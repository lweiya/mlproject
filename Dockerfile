ROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/
COPY ./zh_ner_tender /app/zh_ner_tender
WORKDIR /app/zh_ner_tender
RUN pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pydantic==1.8.2 fastapi==0.61.2 sacremoses transformers tokenizers aiofiles flask_restx uvicorn==0.11.8 spacy-alignments torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python setup.py install
COPY ./scripts /app/app