import torch
from fastapi import FastAPI
from utils import hisab_ner
from model import HisabNerBertModel

app = FastAPI()


@app.get("/")
async def index():
    return {"Working": True}


@app.post("/hisab_ner/")
async def name_entiry_recognition(sentence: str, checkpoint_path: str):

    model = HisabNerBertModel()
    model = torch.load(checkpoint_path)  # load save model
    # print(model)

    return hisab_ner(model, sentence)
