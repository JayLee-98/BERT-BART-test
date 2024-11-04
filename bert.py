# testing

# embedding
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# 원하는 BERT 모델 설정 (예: bert-base-uncased)
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 로컬 경로에 모델 저장
# local_path = "./bert_model"
# tokenizer.save_pretrained(local_path)
# model.save_pretrained(local_path)

@app.get("/")
def read_root():
    return {"message": "BERT 모델이 성공적으로 로드되었습니다."}

@app.post("/get_embedding")
def get_embedding(text):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors="pt")
    
    # BERT 모델을 사용해 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 첫 번째 토큰의 임베딩 벡터 (CLS 토큰)
    embedding = outputs.last_hidden_state[0][0]
    print("embedding: ", embedding)
    return embedding