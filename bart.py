# # 
# from fastapi import FastAPI, Request
# from transformers import pipeline
# from pydantic import BaseModel

# # FastAPI 인스턴스 생성
# app = FastAPI()

# # BERT 모델 초기화
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# @app.on_event("startup")
# async def load_model():
#     # 서버 시작 시 모델을 로드하여 메모리에 유지
#     global summarizer
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     print("Bart summarizer loaded and ready.")

# @app.post("/summarize")
# async def summarize(request: Request):
#     request = """
#     New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
# A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
# Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
# In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
# Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
# 2010 marriage license application, according to court documents.
# Prosecutors said the marriages were part of an immigration scam.
# On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
# After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
# Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
# All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
# Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
# Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
# The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
# Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
# Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
# If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
#     """
#     summary = summarizer(request, max_length=130, min_length=30, do_sample=False)
#     print(summary)
#     return {"summary": summary}

from fastapi import FastAPI, Request
from transformers import pipeline, BartTokenizer, BartModel
from pydantic import BaseModel
import torch

# FastAPI 인스턴스 생성
app = FastAPI()

# BERT 모델 초기화
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# BART 임베딩 모델과 토크나이저 초기화
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartModel.from_pretrained("facebook/bart-large-cnn")

@app.on_event("startup")
async def load_model():
    global summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Bart summarizer loaded and ready.")
    print("BART embedding model loaded and ready.")

# 요약 엔드포인트
@app.post("/summarize")
async def summarize(request: Request):
    text = """
    New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    print(summary)
    return {"summary": summary}

# 임베딩 엔드포인트
@app.post("/embed")
async def embed_text(text: str):
    # 입력 텍스트를 토크나이징하여 모델 입력으로 변환
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    # 인코더에서 임베딩 추출
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    
    # [CLS]에 해당하는 첫 번째 토큰의 벡터를 사용하여 문장 임베딩 생성
    sentence_embedding = encoder_outputs.last_hidden_state[:, 0, :]
    
    # 임베딩을 리스트 형태로 변환하여 반환
    embedding_list = sentence_embedding.squeeze().tolist()
    print("Embedding:", embedding_list)
    return {"embedding": embedding_list}
