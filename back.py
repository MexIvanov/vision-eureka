import uuid
import stamina
import torch
import time
import numpy as np
import pathlib

from PIL import Image
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor

from qdrant_client import QdrantClient
from qdrant_client.http import models

from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path
from docx2pdf import convert as docx_to_pdf


COLLECTION_NAME = "documents_v2"

def get_uuid():
    return str(uuid.uuid4())


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# Initialize ColPali model and processor
model_name = (
    "vidore/colpali-v1.2" # Use the latest version available
)
# TODO: add check to load ColPali on demand
colpali_model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
    quantization_config=BNB_CONFIG,
)
colpali_processor = ColPaliProcessor.from_pretrained(
    "vidore/colpaligemma-3b-pt-448-base",
    quantization_config=BNB_CONFIG,
)

qdrant_client = QdrantClient(url="http://localhost:6333") #TODO: change later



exists = qdrant_client.collection_exists(COLLECTION_NAME)
if not exists:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        on_disk_payload=True,  # store the payload on disk
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            on_disk=True, # move original vectors to disk
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True  # keep only quantized vectors in RAM
                ),
            ),
        ),
    )
   
    
@stamina.retry(on=Exception, attempts=5) # retry mechanism if an exception occurs during the operation
def upsert_to_qdrant(points):
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True


import subprocess
import os

def convert_docx_to_pdf(docx_file, output_dir=None):
    """
    Конвертирует файл DOCX в PDF с использованием LibreOffice.

    :param docx_file: Путь к файлу DOCX.
    :param output_dir: Путь к директории, куда сохранить PDF. Если None, сохраняется в ту же директорию.
    :return: Путь к созданному PDF.
    """
    if not os.path.isfile(docx_file):
        raise FileNotFoundError(f"Файл {docx_file} не найден.")

    if output_dir is None:
        output_dir = os.path.dirname(docx_file)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(docx_file))[0] + '.pdf')

    try:
        subprocess.run(
            ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, docx_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ошибка при конвертации: {e.stderr.decode('utf-8')}") from e

def index_docs(images, metadata, batch_size=4): # Adjust batch_size based on your GPU memory constraints
    # Use tqdm to create a progress bar
    with tqdm(total=len(images), desc="Indexing Progress") as pbar:
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size] # The images are PIL Image objects, so we can use them directly
            metadata_batch = metadata[i : i + batch_size]
            
            # Process and encode images
            with torch.no_grad():
                batch_images = colpali_processor.process_images(batch).to(
                    colpali_model.device
                )
                image_embeddings = colpali_model(**batch_images)

            # Prepare points for Qdrant
            points = []
            for image, mdata, embedding in zip(batch, metadata_batch, image_embeddings):
                # Convert the embedding to a list of vectors
                multivector = embedding.cpu().float().numpy().tolist()

                points.append(
                    models.PointStruct(
                        id=mdata["img_id"],  # we just use the index as the ID
                        vector=multivector,  # This is now a list of vectors
                        payload={
                            "source": mdata["source"],
                        },  # can also add other metadata/data
                    )
                )

            # Upload points to Qdrant
            try:
                upsert_to_qdrant(points)
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue

            # Update the progress bar
            pbar.update(batch_size)

    qdrant_client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),

    )
    print("Indexing complete!")


def query_vdb(query_text):
    with torch.no_grad():
        batch_query = colpali_processor.process_queries([query_text]).to(
            colpali_model.device
        )
        query_embedding = colpali_model(**batch_query)

    #print(query_embedding)
    multivector_query = query_embedding[0].cpu().float().numpy().tolist()

    start_time = time.time()
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=multivector_query,
        limit=10,
        timeout=100,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        )
    )
    end_time = time.time()
    # Search in Qdrant
    print(search_result.points)

    elapsed_time = end_time - start_time
    print(f"Search completed in {elapsed_time:.4f} seconds")
    return search_result.points

def resize_image_to_resolution(img, resolution):
    # Open the image

    # Calculate the current aspect ratio
    aspect_ratio = img.width / img.height
    
    # Determine the new dimensions
    new_width = resolution
    new_height = int(resolution / aspect_ratio)

    # Resize the image
    return img.resize((new_width, new_height))


def convert_files(files):
    images = []
    metadata = []
    for f in tqdm(files):
        path = pathlib.Path(f)
        extension = path.suffix
        print(extension)
        
        if extension in [".docx", ".doc"]:
            new_path = path.with_suffix(".pdf")
            print(new_path)
            try:
                docx_to_pdf(f, new_path)
                f = new_path
            except NotImplementedError:
                convert_docx_to_pdf(f, "RAG")
                f = join("./RAG", new_path)
                pass

            

        new_imgs = convert_from_path(f, dpi=200, thread_count=64)
        images.extend(new_imgs)
        
        for img in new_imgs:
            img_id = get_uuid()
            img_path = (f"imgs/{img_id}.png")

            #img = resize_image_to_resolution(img, 1280)
            img.save(img_path)
            metadata.append({"img_id": img_id, "source": f.split("/")[-1]})


    if (len(images) != len(metadata)):
        return print("Metadata serialization error!")
    else:
        print("Metadata serialization ok!")
    
    return images, metadata


def file_to_vdb(filepaths: list):
    print("Converting files")
    images, metadata = convert_files(filepaths)
    index_docs(images, metadata, batch_size=10)

def reindex_doc_folder():
    onlyfiles = [f for f in listdir("RAG") if isfile(join("RAG", f))]
    onlyfiles2 = []
    #onlyfiles = [onlyfiles[0]] #TODO:REMOVE
    for i in onlyfiles:
        fpath = "./RAG/" + i
        onlyfiles2.append(fpath)
        #if isfile(fpath):
        #    print(fpath)
    print(onlyfiles2)
    file_to_vdb(onlyfiles2)


def load_vlm():
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.


    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        #attn_implementation="flash_attention_2", #doesn't work on zerogpu WTF?!
        trust_remote_code=True,
        #quantization_config=BNB_CONFIG, 
        torch_dtype=torch.bfloat16,
	device_map="auto") #.(to_cuda:0")

    # default processer
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    #processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels) #, quantization_config=BNB_CONFIG)
    return model, processor

def vlm_inference(model, processor, images, text):
   
    img_list = [{"type": "image", "image": Image.open(image)} for image in images]
    img_list.append({"type": "text", "text": text})
    
    #messages = []
    #content = []
    #for img in images: 
    #    content.append({'image': Image.open(img)})
    
    #content.append({'text': text})
    #messages.append({'role': 'user', 'content': content})
    #messages.append({'role': 'assistant', 'content': [{'text': a}]})
    #content = []

    messages = [
        {
            "role": "user",
            "content": img_list,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)#, repetition_penalty=1.5 )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del model
    del processor
    torch.cuda.empty_cache()
    return output_text[0]

#print(str(results[0].id))
#breakpoint()

"""
reindex_doc_folder() - fully reindex RAG folder
file_to_vdb(filepaths) - index list of files
query_vdb("query") - get relevant image metadata from qdrant vdb
"""
#file_to_vdb(["СП_496_1325800_2020_Основания_и_фундаменты_зданий_и_сооружений.docx", "2022_Annual_Report_of_PJSC_MMC_Norilsk_Nickel_rus.pdf"])
#reindex_doc_folder()

"""
PROMPT = "What are the key findings of Global Metals and Mining Outlook"
results = query_vdb(PROMPT)
model, processor = load_vlm()

for idx, res in enumerate(results):
    img = f"imgs/{str(res.id)}.png"
    print(idx, ") | ", str(res.id), " | ", vlm_inference(model, processor, [img], PROMPT))
"""
