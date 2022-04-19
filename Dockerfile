FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/
RUN pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pydantic==1.8.2 fastapi>=0.65.2 sacremoses transformers tokenizers aiofiles flask_restx uvicorn==0.11.8 spacy-alignments torch -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY ./scripts /app/app