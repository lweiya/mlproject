from typing import List, Dict, Any
from enum import Enum

from flask_restx import Model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.tokens import Doc


# 这是api使用的模型
class ModelName(str, Enum):
    # Enum of the available models. This allows the API to raise a more specific
    # error if an invalid model is provided.
    zh_ner_tender = "zh_ner_tender"


DEFAULT_MODEL = ModelName.zh_ner_tender
path = '../training/model-best'
nlp_model = spacy.load(path)



class RequestModel(BaseModel):
    # 定义一个类，用于接收前端传来的数据
    text: str


def get_data(doc: Doc) -> Dict[str, Any]:
    # 将数据提取出来，并返回
    ents = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"text": doc.text, "ents": ents}


# 设置FastAPI，并定义路由
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/process/", summary="Process text")
def process_text(query: RequestModel):
    nlp = nlp_model
    text = query.text
    doc = nlp(text)
    # response_body.append(get_data(doc))
    # return {response_body}
    return (get_data(doc))


