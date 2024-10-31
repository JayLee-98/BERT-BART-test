# multi-lan

from transformers import AutoTokenizer, AutoModelForMaskedLM
from fastapi import FastAPI, Request

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")

@app.on_event("startup")
async def load_model():
    print("xlm-roberta-base summarization model loaded and ready.")

@app.post("/xlm-roberta-base")
async def summarize(text: Request):
    # prepare input
    text = "과거를 떠올려보자. 방송을 보던 우리의 모습을. 독보적인 매체는 TV였다."
    encoded_input = tokenizer(text, return_tensors='pt')

    # forward pass
    output = model(**encoded_input)
    print("outuput", output)