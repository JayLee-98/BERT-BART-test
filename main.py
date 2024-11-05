from fastapi import FastAPI, APIRouter, Request
from pydantic import BaseModel
import torch
from typing import List, Optional

# 모델 임포트
from transformers import (
    PreTrainedTokenizerFast, 
    BartForConditionalGeneration,
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    AutoTokenizer, 
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartModel,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="NLP Models API")

# Request 모델 정의
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

# 라우터 설정
kobart_router = APIRouter(prefix="/kobart", tags=["KoBART"])
kobart_v3_router = APIRouter(prefix="/kobart-v3", tags=["KoBART-v3"])
kot5_router = APIRouter(prefix="/kot5", tags=["KoT5"])
sentence_transformer_router = APIRouter(prefix="/sentence-transformer", tags=["Sentence Transformer"])
pegasus_router = APIRouter(prefix="/pegasus", tags=["Pegasus"])
mt5_router = APIRouter(prefix="/mt5", tags=["mT5"])
xlm_roberta_router = APIRouter(prefix="/xlm-roberta", tags=["XLM-RoBERTa"])
t5_router = APIRouter(prefix="/t5", tags=["T5"])
bert_router = APIRouter(prefix="/bert", tags=["BERT"])
bart_router = APIRouter(prefix="/bart", tags=["BART"])

# KoBART 라우터
@kobart_router.post("/summarize")
async def kobart_summarize(request: TextRequest):
    input_ids = kobart_tokenizer.encode(request.text)
    input_ids = [kobart_tokenizer.bos_token_id] + input_ids + [kobart_tokenizer.eos_token_id]
    
    summary_ids = kobart_model.generate(
        torch.tensor([input_ids]).to(device),
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = kobart_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    return {"summary": summary}

@kobart_router.post("/batch-summarize")
async def kobart_batch_summarize(request: BatchTextRequest):
    summaries = []
    for text in request.texts:
        input_ids = kobart_tokenizer.encode(text)
        input_ids = [kobart_tokenizer.bos_token_id] + input_ids + [kobart_tokenizer.eos_token_id]
        summary_ids = kobart_model.generate(
            torch.tensor([input_ids]).to(device),
            max_length=150,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = kobart_tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summaries.append(summary)
    return {"summaries": summaries}

# KoBART-v3 라우터
@kobart_v3_router.post("/summarize")
async def kobart_v3_summarize(request: TextRequest):
    input_ids = kobart_v3_tokenizer(request.text, return_tensors="pt")["input_ids"]
    summary_ids = kobart_v3_model.generate(
        input_ids,
        max_length=128,
        min_length=32,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    summary = kobart_v3_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# KoT5 라우터
@kot5_router.post("/summarize")
async def kot5_summarize(request: TextRequest):
    summary = kot5_summarizer(request.text, max_length=150, min_length=40)
    return {"summary": summary[0]["summary_text"]}

# Sentence Transformer 라우터
@sentence_transformer_router.post("/embed")
async def sentence_transformer_embed(request: TextRequest):
    embedding = sentence_transformer_model.encode(request.text)
    return {"embedding": embedding.tolist()}

@sentence_transformer_router.post("/similarity")
async def sentence_transformer_similarity(texts: BatchTextRequest):
    embeddings = sentence_transformer_model.encode(texts.texts)
    similarity_matrix = cosine_similarity(embeddings)
    return {"similarity_matrix": similarity_matrix.tolist()}

# Pegasus 라우터
@pegasus_router.post("/summarize")
async def pegasus_summarize(request: TextRequest):
    inputs = pegasus_tokenizer(request.text, truncation=True, padding="longest", return_tensors="pt").to(device)
    summary_ids = pegasus_model.generate(**inputs)
    summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# mT5 라우터
@mt5_router.post("/summarize")
async def mt5_summarize(request: TextRequest):
    input_ids = mt5_tokenizer(
        request.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]
    
    output_ids = mt5_model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    
    summary = mt5_tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return {"summary": summary}

# XLM-RoBERTa 라우터
@xlm_roberta_router.post("/analyze")
async def xlm_roberta_analyze(request: TextRequest):
    encoded_input = xlm_roberta_tokenizer(request.text, return_tensors='pt')
    output = xlm_roberta_model(**encoded_input)
    return {"logits": output.logits.tolist()}

# T5 라우터
@t5_router.post("/summarize")
async def t5_summarize(request: TextRequest):
    summary = t5_summarizer(request.text, max_length=130, min_length=30)
    return {"summary": summary[0]["summary_text"]}

# BERT 라우터
@bert_router.post("/embed")
async def bert_embed(request: TextRequest):
    inputs = bert_tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state[0][0].tolist()
    return {"embedding": embedding}

@bert_router.post("/batch-embed")
async def bert_batch_embed(request: BatchTextRequest):
    embeddings = []
    for text in request.texts:
        inputs = bert_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[0][0].tolist()
        embeddings.append(embedding)
    return {"embeddings": embeddings}

# BART 라우터
@bart_router.post("/summarize")
async def bart_summarize(request: TextRequest):
    summary = bart_summarizer(request.text, max_length=130, min_length=30)
    return {"summary": summary[0]["summary_text"]}

@bart_router.post("/embed")
async def bart_embed(request: TextRequest):
    inputs = bart_tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = bart_model.encoder(**inputs)
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return {"embedding": sentence_embedding}

# 모든 라우터를 앱에 포함
app.include_router(kobart_router)
app.include_router(kobart_v3_router)
app.include_router(kot5_router)
app.include_router(sentence_transformer_router)
app.include_router(pegasus_router)
app.include_router(mt5_router)
app.include_router(xlm_roberta_router)
app.include_router(t5_router)
app.include_router(bert_router)
app.include_router(bart_router)

# 모델 로드
@app.on_event("startup")
async def load_models():
    global device, kobart_tokenizer, kobart_model, kobart_v3_tokenizer, kobart_v3_model
    global kot5_summarizer, sentence_transformer_model, pegasus_tokenizer, pegasus_model
    global mt5_tokenizer, mt5_model, xlm_roberta_tokenizer, xlm_roberta_model
    global t5_summarizer, bert_tokenizer, bert_model, bart_summarizer, bart_tokenizer, bart_model

    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # KoBART
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    kobart_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization').to(device)
    
    # KoBART-v3
    kobart_v3_tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
    kobart_v3_model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3").to(device)
    
    # KoT5
    kot5_summarizer = pipeline("summarization", model="psyche/KoT5-summarization")
    
    # Sentence Transformer
    sentence_transformer_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Pegasus
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
    
    # mT5
    mt5_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    mt5_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum").to(device)
    
    # XLM-RoBERTa
    xlm_roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    xlm_roberta_model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base").to(device)
    
    # T5
    t5_summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
    
    # BERT
    bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    
    # BART
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = BartModel.from_pretrained("facebook/bart-large-cnn").to(device)
    
    print("All models loaded successfully!")

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "NLP Models API is running",
        "available_models": [
            "KoBART", "KoBART-v3", "KoT5", "Sentence Transformer", 
            "Pegasus", "mT5", "XLM-RoBERTa", "T5", "BERT", "BART"
        ]
    }