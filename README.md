# RAG Local

Локальный RAG-ассистент с веб-интерфейсом на Streamlit. Поддерживает два режима работы:

- **Simple Chat** -- обычный чат с LLM через OpenRouter.
- **RAG Chat** -- загрузка PDF-документов, индексация через Pinecone и ответы на вопросы по содержимому документов.

Стек: LangChain, OpenAI Embeddings, Pinecone, Streamlit.

## Переменные окружения

Скопируйте `.env.sample` в `.env` и заполните собственными API-ключами и названием индекса в Pinecone.

> [!NOTE]
> Индекс в Pinecone будет создан автоматически при первой загрузке документов.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
streamlit run src/app.py
```


## Запуск в Docker

```bash
docker build -t rag-local .
docker run -p 8501:8501 --env-file .env rag-local
```

Приложение будет доступно по адресу `http://localhost:8501`.
