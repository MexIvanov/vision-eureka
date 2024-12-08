"""
Код для векторизации и поиска документов

Данный код предназначен для обработки, векторизации и поиска данных в документах
с использованием методов обработки естественного языка (NLP) и анализа изображений.
Код интегрирован с Qdrant — системой поиска векторов, а также с моделью ColPali 
для извлечения эмбеддингов из документов.

Модули:
    - `get_uuid`: Генерация уникального идентификатора для изображений.
    - `upsert_to_qdrant`: Загрузка данных в Qdrant с механизмом повторных попыток.
    - `convert_docx_to_pdf`: Конвертация файлов DOCX в PDF с использованием LibreOffice.
    - `index_docs`: Индексация изображений в Qdrant пакетами.
    - `query_vdb`: Поиск релевантных данных в Qdrant на основе эмбеддингов.
    - `resize_image_to_resolution`: Изменение размера изображений с сохранением пропорций.
    - `convert_files`: Конвертация документов (DOCX, PDF) в изображения для индексации.
    - `file_to_vdb`: Индексация списка файлов в Qdrant.
    - `reindex_doc_folder`: Полная переиндексация папки "RAG".
    - `load_vlm`: Загрузка vision-language модели (Qwen2VL).
    - `vlm_inference`: Проведение инференса с изображениями и текстовыми запросами с использованием Qwen2VL.

Инициализация клиента Qdrant:
    Клиент Qdrant настраивается для взаимодействия с базой данных.

Информация о моделях:
    - ColPali: Используется для обработки и создания эмбеддингов из изображений документов.
    - Qwen2VL: Применяется для инференса на основе изображений и текста.

Примеры использования:
    - Для индексации файлов:
        `file_to_vdb(["file1.docx", "file2.pdf"])`
    - Для переиндексации папки:
        `reindex_doc_folder()`
    - Для поиска по базе:
        `results = query_vdb("Ваш запрос")`
    - Для инференса vision-language:
        ```
        model, processor = load_vlm()
        output = vlm_inference(model, processor, ["path_to_image.png"], "Ваш запрос")
        ```

Зависимости:
    - PyTorch
    - Transformers
    - PIL (Pillow)
    - Qdrant Client
    - Subprocess (для конвертации DOCX в PDF через LibreOffice)
    - pdf2image (для конвертации PDF в изображения)
"""

import uuid
import stamina
import torch
import time
import numpy as np
import pathlib
import os
import subprocess

from PIL import Image
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
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
    """
    Генерация уникального идентификатора.
    """
    return str(uuid.uuid4())

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Инициализация модели ColPali и процессора
model_name = "vidore/colpali-v1.2"
colpali_model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    quantization_config=BNB_CONFIG,
)
colpali_processor = ColPaliProcessor.from_pretrained(
    "vidore/colpaligemma-3b-pt-448-base",
    quantization_config=BNB_CONFIG,
)

qdrant_client = QdrantClient(url="http://localhost:6333")

# Проверка существования коллекции и её создание, если она отсутствует
exists = qdrant_client.collection_exists(COLLECTION_NAME)
if not exists:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            on_disk=True,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True
                ),
            ),
        ),
    )

@stamina.retry(on=Exception, attempts=5)
def upsert_to_qdrant(points):
    """
    Загрузка данных в Qdrant с повторными попытками в случае ошибки.
    """
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return False
    return True

def convert_docx_to_pdf(docx_file, output_dir=None):
    """
    Конвертирует DOCX файл в PDF с использованием LibreOffice.
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

def index_docs(images, metadata, batch_size=4):
    """
    Индексация изображений и метаданных в Qdrant пакетами.
    """
    with tqdm(total=len(images), desc="Индексация прогресс") as pbar:
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            metadata_batch = metadata[i:i + batch_size]
            
            with torch.no_grad():
                batch_images = colpali_processor.process_images(batch).to(colpali_model.device)
                image_embeddings = colpali_model(**batch_images)

            points = []
            for image, mdata, embedding in zip(batch, metadata_batch, image_embeddings):
                multivector = embedding.cpu().float().numpy().tolist()
                points.append(
                    models.PointStruct(
                        id=mdata["img_id"],
                        vector=multivector,
                        payload={"source": mdata["source"]},
                    )
                )

            upsert_to_qdrant(points)
            pbar.update(batch_size)

    qdrant_client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
    )
    print("Индексация завершена!")

def query_vdb(query_text):
    """
    Выполняет запрос в Qdrant для поиска наиболее релевантных данных на основе эмбеддингов.
    """
    with torch.no_grad():
        batch_query = colpali_processor.process_queries([query_text]).to(colpali_model.device)
        query_embedding = colpali_model(**batch_query)

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

    print(f"Запрос выполнен за {end_time - start_time:.4f} секунд")
    return search_result.points

def resize_image_to_resolution(img, resolution):
    """
    Изменяет размер изображения до заданного разрешения с сохранением пропорций.
    """
    aspect_ratio = img.width / img.height
    new_width = resolution
    new_height = int(resolution / aspect_ratio)
    return img.resize((new_width, new_height))

def convert_files(files):
    """
    Конвертирует файлы DOCX и PDF в изображения, сохраняет метаданные.
    """
    images = []
    metadata = []
    for f in tqdm(files, desc="Обработка файлов"):
        path = pathlib.Path(f)
        extension = path.suffix
        
        # Проверяем существование файла
        if not path.exists():
            print(f"Файл не найден: {f}")
            continue

        if extension in [".docx", ".doc"]:
            new_path = path.with_suffix(".pdf")
            try:
                # Конвертация DOCX -> PDF
                docx_to_pdf(str(path), str(new_path))
                f = new_path
            except Exception as e:
                print(f"Ошибка конвертации {f}: {e}")
                continue

        # Конвертация PDF -> изображения
        try:
            new_imgs = convert_from_path(f, dpi=200, thread_count=4)
            images.extend(new_imgs)
        except Exception as e:
            print(f"Ошибка обработки PDF {f}: {e}")
            continue
        
        for img in new_imgs:
            img_id = get_uuid()
            img_path = f"imgs/{img_id}.png"
            img.save(img_path)
            metadata.append({"img_id": img_id, "source": f.split("/")[-1]})

    if len(images) != len(metadata):
        raise ValueError("Ошибка сериализации метаданных!")
    
    print("Сериализация метаданных успешно завершена.")
    return images, metadata


def file_to_vdb(filepaths):
    """
    Конвертирует и индексирует список файлов в базу данных Qdrant.
    """
    print("Начало конвертации файлов")
    images, metadata = convert_files(filepaths)
    index_docs(images, metadata, batch_size=10)

def reindex_doc_folder():
    """
    Полная переиндексация папки "RAG".
    """
    onlyfiles = [f for f in listdir("RAG") if isfile(join("RAG", f))]
    onlyfiles_paths = [f"./RAG/{file}" for file in onlyfiles]
    file_to_vdb(onlyfiles_paths)

def load_vlm():
    """
    Загружает модель Qwen2VL и процессор.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    return model, processor

def vlm_inference(model, processor, images, text):
    """
    Выполняет инференс на изображениях и текстовом запросе с использованием Qwen2VL.
    """
    img_list = [{"type": "image", "image": Image.open(image)} for image in images]
    img_list.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": img_list}]

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
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del model, processor
    torch.cuda.empty_cache()
    return output_text[0]
