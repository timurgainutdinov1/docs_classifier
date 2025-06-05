import logging
import os
import uuid

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_gigachat.chat_models import GigaChat


@st.cache_data
def load_default_prompt():
    """
    Загружает шаблон стандартного запроса.
    """
    with open("default_prompt.txt", "r", encoding="utf-8") as file:
        text = file.read()

    return text


def save_file(file):
    """
    Сохраняет загруженный файл с уникальным идентификатором.

    Args:
        file: Объект загруженного файла.

    Returns:
        str: Имя сохраненного файла с уникальным идентификатором.
    """
    unique_id = str(uuid.uuid4())
    file_name = f"{unique_id}_{file.name}"
    with open(file_name, "wb") as f:
        f.write(file.getbuffer())
    return file_name


def extract_text_from_file(uploaded_file: str) -> str:
    """
    Извлекает текстовое содержимое из файлов PDF, DOCX.

    Args:
        uploaded_file (str): Путь к загруженному файлу.

    Returns:
        str: Извлеченный текст из файла. Возвращает пустую строку,
             если формат файла не поддерживается
             или произошла ошибка при чтении.

    Поддерживаемые форматы файлов:
        - .pdf: Использует PyPDFLoader для извлечения текста со всех страниц.
        - .docx: Использует Docx2txtLoader для извлечения текстового
                 содержимого.
    """
    try:
        if uploaded_file.endswith(".docx"):
            return Docx2txtLoader(uploaded_file).load()[0].page_content
        elif uploaded_file.endswith(".pdf"):
            return PyPDFLoader(uploaded_file, mode="single").load()[0].page_content
        else:
            logging.error(f"Неподдерживаемый формат файла: {uploaded_file}")
            raise ValueError(f"Неподдерживаемый формат файла: {uploaded_file}")
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {uploaded_file}: {str(e)}")
        st.error(f"Ошибка при чтении файла {uploaded_file}: {str(e)}")


def get_model(model_name: str) -> str:
    """
    Возвращает модель GigaChat в зависимости от выбранного имени модели.

    Args:
        model_name (str): Имя модели GigaChat.

    Returns:
        str: Имя модели API GigaChat.
    """
    models = {
        "GigaChat-Lite": "GigaChat",
        "GigaChat-Pro": "GigaChat-Pro",
        "GigaChat-Max": "GigaChat-Max",
    }
    return models[model_name]


def main():

    logging.basicConfig(level=logging.INFO)

    st.title("Интеллектуальный помощник для классификации документов")

    document = st.file_uploader(
        "📄 Загрузите документ для анализа (PDF/DOCX)", type=["pdf", "docx"]
    )

    default_prompt = load_default_prompt()

    prompt = st.text_area("✏️ Введите текст запроса", value=default_prompt, height=400)

    api_key = st.text_input("Введите API ключ для работы с GigaChat", type="password")

    model_name = st.selectbox(
        "Выберите модель GigaChat",
        [
            "GigaChat-Lite ⚡",
            "GigaChat-Pro ⚡⚡",
            "GigaChat-Max ⚡⚡⚡",
        ],
        index=0,
    )

    scope = st.selectbox(
        "Выберите версию API",
        [
            "GIGACHAT_API_PERS (для физических лиц)",
            "GIGACHAT_API_CORP",
            "GIGACHAT_API_B2B",
        ],
        index=0,
    ).split(" ")[0]

    model = get_model(model_name.split(" ")[0])

    start_button = st.button("Анализировать документ")

    if start_button:
        if document and api_key:
            document_path = save_file(document)

            document_text = extract_text_from_file(document_path)

            template = PromptTemplate.from_template(prompt)

            llm = GigaChat(
                model=model,
                credentials=api_key,
                scope=scope,
                temperature=0,
                verify_ssl_certs=False,
            )

            chain = template | llm | StrOutputParser()

            try:
                result = chain.invoke({"document_text": document_text})
                st.header("📊 Результат анализа документа")
                st.markdown(result)
            except Exception as e:
                st.error(f"Не удалось выполнить анализ документа. Ошибка: {e}")
            finally:
                os.remove(document_path)
        else:
            st.error(
                "Для анализа документа необходимо загрузить файл и ввести API ключ"
            )


if __name__ == "__main__":
    main()
