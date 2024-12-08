"""
Этот код реализует веб-интерфейс на основе Gradio для анализа документов, изображений и выполнения текстового поиска. 
Основные функции системы включают:

1. Поиск в базе данных документов (Qdrant).
2. Анализ изображений с помощью .
3. Добавление новых документов в базу данных.

Модули и функции:
- Gradio: используется для создания пользовательского интерфейса.
- query_vdb: выполняет текстовый поиск в базе данных Qdrant.
- file_to_vdb: добавляет документы в базу данных.
- vlm_inference: анализирует текст и изображения с использованием модели Qwen.
- load_vlm: загружает модель Qwen и процессор.
"""

import gradio as gr
from back import query_vdb, file_to_vdb, vlm_inference, load_vlm  # Импорт функций из back.py

"""
Загрузка модели и процессора для анализа текста и изображений:
- model: модель Qwen для генерации ответов.
- processor: процессор для предварительной обработки данных.
"""
model, processor = load_vlm()

"""
Функция analyze_image_ui выполняет анализ загруженного изображения на основе текста-запроса.
- image: путь к загруженному изображению.
- text: текстовый запрос для анализа изображения.
Возвращает ответ модели Qwen на основе предоставленных данных.
"""
def analyze_image_ui(image, text):
    if not image:
        return "Изображение не предоставлено."
    img_path = image.name  # Получаем путь к загруженному изображению
    response = vlm_inference(model, processor, [img_path], text)
    return response

"""
Функция query_vdb_ui выполняет текстовый поиск в базе данных Qdrant:
- query: текстовый запрос для поиска в базе данных.
Возвращает:
- img_path: путь к найденному изображению.
- source: источник (название документа).
- qwen_answer: ответ модели Qwen на основе текста-запроса.
"""
def query_vdb_ui(query):
    results = query_vdb(query)
    if not results:
        return None, "Документы не найдены.", None

    top_result = results[0]
    img_id = str(top_result.id)
    source = top_result.payload.get("source", "Неизвестный источник")
    img_path = f"imgs/{img_id}.png"
    qwen_answer = vlm_inference(model, processor, [img_path], query)

    return img_path, source, qwen_answer

"""
Функция index_files_ui обрабатывает загрузку документов и добавляет их в базу данных.
- files: список загруженных файлов.
Возвращает статус выполнения (успешно или ошибка).
"""
def index_files_ui(files):
    print(f"Получены файлы: {files}")
    if not files:
        return "Файлы не были загружены. Пожалуйста, загрузите корректные файлы."
    file_paths = [file.name for file in files]
    file_to_vdb(file_paths)
    return "Файлы успешно добавлены в базу данных."

"""
Создание веб-интерфейса с использованием Gradio.
Система имеет следующие вкладки:
- Поиск документов: поиск документов по текстовому запросу.
- Анализ изображения: анализ изображения с помощью .
- Загрузка файлов для индексации: добавление новых документов в базу данных.
- О системе: информация о возможностях системы.
"""
with gr.Blocks(css=".block-title {text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;}") as demo:
    # Заголовок
    gr.Markdown(
        """
        <div class="block-title">📄 Система анализа документов и изображений от MADI Trailblazers</div>
        """
    )

    # Вкладка поиска
    with gr.Tab("🔍 Поиск документов"):
        gr.Markdown("### Поиск документов и получение ответа от системы")
        with gr.Row():
            query = gr.Textbox(label="Введите текстовый запрос", placeholder="Введите запрос для поиска...", lines=2)
            search_button = gr.Button("Найти", variant="primary")

        with gr.Row():
            img_output = gr.Image(label="Соответствующее изображение", type="filepath", interactive=False)
            source_output = gr.Textbox(label="Источник", interactive=False, lines=2)
            qwen_output = gr.Textbox(label="Ответ системы", interactive=False, lines=20)  

        search_button.click(
            query_vdb_ui, 
            inputs=[query], 
            outputs=[img_output, source_output, qwen_output]
        )

    # Вкладка анализа изображения
    with gr.Tab("🖼️ Анализ изображения"):
        gr.Markdown("### Загрузите изображение и введите текстовый запрос для анализа")
        with gr.Row():
            uploaded_image = gr.File(label="Загрузить изображение", file_types=[".png", ".jpg", ".jpeg"])
            image_query = gr.Textbox(label="Введите запрос для анализа изображения", placeholder="Опишите контекст изображения", lines=2)
            analyze_button = gr.Button("Анализировать", variant="primary")

        with gr.Row():
            analysis_output = gr.Textbox(label="Ответ системы", interactive=False, lines=10) 

        analyze_button.click(
            analyze_image_ui,
            inputs=[uploaded_image, image_query],
            outputs=[analysis_output],
        )

    # Вкладка загрузки файлов
    with gr.Tab("📤 Загрузка файлов для индексации"):
        gr.Markdown("### Загрузите документы для обновления базы данных")
        with gr.Row():
            files = gr.Files(label="Загрузите файлы", file_types=[".docx", ".pdf"])
            index_button = gr.Button("Добавить файлы", variant="secondary")
            status_output = gr.Textbox(label="Статус", interactive=False, lines=3)  
        index_button.click(index_files_ui, inputs=[files], outputs=[status_output])

    # Вкладка о системе
    with gr.Tab("ℹ️ О системе"):
        gr.Markdown(
            """
            ### О системе
            - Эта система позволяет искать документы, загружать файлы и анализировать изображения.
            - Вы можете добавлять новые документы, чтобы база данных всегда была актуальной.
            - Построена с использованием Gradio, Qdrant и модели Qwen.
            """
        )

# Запуск интерфейса
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
