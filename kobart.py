from fastapi import FastAPI, Request
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization').to("cuda" if torch.cuda.is_available() else "cpu")

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

@app.on_event("startup")
async def load_model():
    print("Kobart summarization model loaded and ready.")

@app.post("/Summarize")
async def summarize(text: Request):
    # text = "과거를 떠올려보자. 방송을 보던 우리의 모습을. 독보적인 매체는 TV였다. 온 가족이 둘러앉아 TV를 봤다. 간혹 가족들끼리 뉴스와 드라마, 예능 프로그램을 둘러싸고 리모컨 쟁탈전이 벌어지기도  했다. 각자 선호하는 프로그램을 ‘본방’으로 보기 위한 싸움이었다. TV가 한 대인지 두 대인지 여부도 그래서 중요했다. 지금은 어떤가. ‘안방극장’이라는 말은 옛말이 됐다. TV가 없는 집도 많다. 미디어의 혜 택을 누릴 수 있는 방법은 늘어났다. 각자의 방에서 각자의 휴대폰으로, 노트북으로, 태블릿으로 콘텐츠 를 즐긴다."
    text = """
    29일(이하 현지시간) CNN에 따르면 소수의 북한군이 이미 우크라이나 내부로 진입했다는 것이 서방 당국의 판단이다.

CNN은 이날 2명의 서방 정보 당국자를 인용, "소수의 북한군이 이미 우크라이나 내부에 침투했다"면서 "당국자들은 북한군이 러시아 동부에서 훈련을 마치고 최전선으로 이동하게 되면 침투 병력 규모도 늘어날 것으로 보고 있다"고 보도했다.

한 정보당국자는 "상당수의 북한군이 이미 작전 중인 것으로 보인다"고 밝혔다.

미국은 아직 확증하지 못한다는 입장인 것으로 전해지만, 한국 정부에서 파병 사실을 확인한 이후에도 미국에서 이를 인정하기까지 시차를 감안하면 이미 국경을 넘었을 가능성을 배제하기 어려운 게 사실이다.

러시아로 파병된 북한군 가운데 훈련을 마친 상당수가 극비리에 우크라이나 접경지역인 러시아 쿠르스크로 이동한 것으로 전해져 결정만 내린다면 언제든 쿠르스크 전선이나 우크라이나로 국경을 넘을 수 있는 상황으로 보인다.

국가정보원은 전날 비공개로 진행된 국회 정보위 국정감사에서 "김영복 조선인민군 총참모부 부총참모장을 포함한 선발대가 전선으로 이동 중이라는 첩보가 있는데 이에 대해 확인 중"이라고 밝혔다. 국정원은 또 북한이 올해 말까지 러시아에 모두 1만900명을 파병할 전망이라고 보고했다.

볼로디미르 젤렌스키 우크라이나 대통령은 윤석열 대통령과 통화에서 "북한군의 우크라이나 전선 투입이 임박해 있다"며 "이에 따라 전쟁이 지금까지 경험하지 못한 새로운 국면으로 접어들고 있다"고 평가했다.

러시아에 파병된 북한군이 이미 전투에 투입됐으며 우크라이나군과 교전으로 전사자도 발생했다는 우크라이나군 지원단체 주장까지 나왔다.

리투아니아 비영리기구(NGO) '블루-옐로'의 요나스 오만 대표는 28일(현지시간) 현지 매체 LRT에 "우리가 지원하는 우크라이나군 부대와 북한군의 첫 육안 접촉은 10월 25일 쿠르스크에서 이뤄졌다"며 "내가 알기로 한국인(북한군)은 1명 빼고 전부 사망했다. 생존한 1명은 부랴트인이라는 서류를 갖고 있었다"고 말했다.
"""
    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    # summary_ids = model.generate(torch.tensor([input_ids]))
    # print(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True))
    # output: CNN에 따르면 소수의 북한군이 이미 우크라이나 내부에 침투한 것으로

    # 요약 생성
    summary_ids = model.generate(
        torch.tensor([input_ids]).to(model.device), 
        max_length=150,  # 더 긴 요약을 위해 max_length 증가
        min_length=50,   # 최소 길이 설정
        num_beams=4,     # Beam search 사용
        length_penalty=2.0,
        early_stopping=True
    )
    
    summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    print("summary_text", summary_text)
    # summary_text CNN에 따르면 소수의 북한군이 이미 우크라이나 내부로 진입했다는 것이 서방 당국의 판단이며, 
    # 러시아에 파병된 북한군 가운데 훈련을 마친 상당수가 극비리에 우크라이나 접경지역인 러시아 쿠르스크로 이동한 것으로 전해져 결정만 내린다면 언제든 쿠르스크 전선이나 우크라이나로 국경을 넘을 수 있는 상황이다.
    return {"summary": summary_text}

@app.post("/SummarizeAndEmbed")
async def summarize_and_embed(src_text: Request):
    # 입력 텍스트 설정
    src_text = """
29일(이하 현지시간) CNN에 따르면 소수의 북한군이 이미 우크라이나 내부로 진입했다는 것이 서방 당국의 판단이다.

CNN은 이날 2명의 서방 정보 당국자를 인용, "소수의 북한군이 이미 우크라이나 내부에 침투했다"면서 "당국자들은 북한군이 러시아 동부에서 훈련을 마치고 최전선으로 이동하게 되면 침투 병력 규모도 늘어날 것으로 보고 있다"고 보도했다.

한 정보당국자는 "상당수의 북한군이 이미 작전 중인 것으로 보인다"고 밝혔다.

미국은 아직 확증하지 못한다는 입장인 것으로 전해지만, 한국 정부에서 파병 사실을 확인한 이후에도 미국에서 이를 인정하기까지 시차를 감안하면 이미 국경을 넘었을 가능성을 배제하기 어려운 게 사실이다.

러시아로 파병된 북한군 가운데 훈련을 마친 상당수가 극비리에 우크라이나 접경지역인 러시아 쿠르스크로 이동한 것으로 전해져 결정만 내린다면 언제든 쿠르스크 전선이나 우크라이나로 국경을 넘을 수 있는 상황으로 보인다.

국가정보원은 전날 비공개로 진행된 국회 정보위 국정감사에서 "김영복 조선인민군 총참모부 부총참모장을 포함한 선발대가 전선으로 이동 중이라는 첩보가 있는데 이에 대해 확인 중"이라고 밝혔다. 국정원은 또 북한이 올해 말까지 러시아에 모두 1만900명을 파병할 전망이라고 보고했다.

볼로디미르 젤렌스키 우크라이나 대통령은 윤석열 대통령과 통화에서 "북한군의 우크라이나 전선 투입이 임박해 있다"며 "이에 따라 전쟁이 지금까지 경험하지 못한 새로운 국면으로 접어들고 있다"고 평가했다.

러시아에 파병된 북한군이 이미 전투에 투입됐으며 우크라이나군과 교전으로 전사자도 발생했다는 우크라이나군 지원단체 주장까지 나왔다.

리투아니아 비영리기구(NGO) '블루-옐로'의 요나스 오만 대표는 28일(현지시간) 현지 매체 LRT에 "우리가 지원하는 우크라이나군 부대와 북한군의 첫 육안 접촉은 10월 25일 쿠르스크에서 이뤄졌다"며 "내가 알기로 한국인(북한군)은 1명 빼고 전부 사망했다. 생존한 1명은 부랴트인이라는 서류를 갖고 있었다"고 말했다.
"""

    # 입력 텍스트 인코딩
    raw_input_ids = tokenizer.encode(src_text, return_tensors="pt").to(model.device)
    input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]).to(model.device), raw_input_ids, torch.tensor([[tokenizer.eos_token_id]]).to(model.device)], dim=-1)

    # 요약 생성
    summary_ids = model.generate(
        input_ids, 
        max_length=150,  # 더 긴 요약을 위해 max_length 증가
        min_length=50,   # 최소 길이 설정
        num_beams=4,     # Beam search 사용
        length_penalty=2.0,
        early_stopping=True
    )
    summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    
    # 임베딩 추출 (encoder의 마지막 레이어 출력 사용)
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids)[0]
        sentence_embedding = encoder_outputs.mean(dim=1).squeeze().cpu().numpy()  # 문장 전체의 평균 임베딩
    
    # 요약과 임베딩 값 반환
    return {
        "summary": summary_text,
        "embedding": sentence_embedding.tolist()  # JSON으로 반환하기 위해 리스트로 변환
    }

@app.post("/SummarizeAndChunk")
async def summarize_and_chunk(src_text: Request):
#     src_text = """
# 29일(이하 현지시간) CNN에 따르면 소수의 북한군이 이미 우크라이나 내부로 진입했다는 것이 서방 당국의 판단이다.

# CNN은 이날 2명의 서방 정보 당국자를 인용, "소수의 북한군이 이미 우크라이나 내부에 침투했다"면서 "당국자들은 북한군이 러시아 동부에서 훈련을 마치고 최전선으로 이동하게 되면 침투 병력 규모도 늘어날 것으로 보고 있다"고 보도했다.

# 한 정보당국자는 "상당수의 북한군이 이미 작전 중인 것으로 보인다"고 밝혔다.

# 미국은 아직 확증하지 못한다는 입장인 것으로 전해지만, 한국 정부에서 파병 사실을 확인한 이후에도 미국에서 이를 인정하기까지 시차를 감안하면 이미 국경을 넘었을 가능성을 배제하기 어려운 게 사실이다.

# 러시아로 파병된 북한군 가운데 훈련을 마친 상당수가 극비리에 우크라이나 접경지역인 러시아 쿠르스크로 이동한 것으로 전해져 결정만 내린다면 언제든 쿠르스크 전선이나 우크라이나로 국경을 넘을 수 있는 상황으로 보인다.

# 국가정보원은 전날 비공개로 진행된 국회 정보위 국정감사에서 "김영복 조선인민군 총참모부 부총참모장을 포함한 선발대가 전선으로 이동 중이라는 첩보가 있는데 이에 대해 확인 중"이라고 밝혔다. 국정원은 또 북한이 올해 말까지 러시아에 모두 1만900명을 파병할 전망이라고 보고했다.

# 볼로디미르 젤렌스키 우크라이나 대통령은 윤석열 대통령과 통화에서 "북한군의 우크라이나 전선 투입이 임박해 있다"며 "이에 따라 전쟁이 지금까지 경험하지 못한 새로운 국면으로 접어들고 있다"고 평가했다.

# 러시아에 파병된 북한군이 이미 전투에 투입됐으며 우크라이나군과 교전으로 전사자도 발생했다는 우크라이나군 지원단체 주장까지 나왔다.

# 리투아니아 비영리기구(NGO) '블루-옐로'의 요나스 오만 대표는 28일(현지시간) 현지 매체 LRT에 "우리가 지원하는 우크라이나군 부대와 북한군의 첫 육안 접촉은 10월 25일 쿠르스크에서 이뤄졌다"며 "내가 알기로 한국인(북한군)은 1명 빼고 전부 사망했다. 생존한 1명은 부랴트인이라는 서류를 갖고 있었다"고 말했다.
# """

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # src_text = """
    # 한국의 경제 상황은 최근 몇 년간 안정적으로 성장하고 있습니다. 2020년 이후, 한국은 빠르게 회복하여 주요 지표에서 긍정적인 성과를 보이고 있습니다. 특히, 수출이 GDP 성장의 주요 원동력으로 작용하고 있습니다.

    # 하지만 고령화와 같은 인구 문제는 여전히 큰 과제입니다. 고령화로 인해 일할 수 있는 인구가 감소하고 있으며, 이에 따라 세수 확보와 복지 부담 문제도 증가하고 있습니다. 이러한 문제를 해결하기 위해 다양한 정책이 필요합니다.

    # 최근에는 인공지능(AI) 기술이 많은 분야에서 적용되고 있습니다. 특히, 의료 분야에서는 AI가 영상 판독과 같은 진단 과정에서 중요한 역할을 하고 있습니다. AI를 활용한 진단 도구는 진단 정확도를 높이고 의료 비용을 절감하는 데 기여하고 있습니다.

    # 미국과 중국 간의 무역 갈등은 여전히 전 세계 경제에 영향을 미치고 있습니다. 특히, 반도체와 같은 핵심 기술 분야에서 두 나라 간의 경쟁이 치열합니다. 이로 인해 여러 국가가 무역 정책을 조정하고 있으며, 한국도 이에 대한 대비책을 고민하고 있습니다.

    # 문화 콘텐츠 수출은 한국 경제에 또 다른 긍정적인 영향을 미치고 있습니다. K-팝, 영화, 드라마 등의 콘텐츠는 전 세계에서 큰 인기를 끌고 있으며, 이를 통해 한국의 문화적 영향력이 확대되고 있습니다. 이런 성장은 관광산업 활성화와도 연결됩니다.
    # """

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    src_text = """
    Anytalk 프로젝트에서 로컬 네트워크와 외부 통신을 보안하기 위해 IP 기반 SSL 인증서를 설정할 필요가 있었습니다. SSL 인증서가 없으면 외부와의 통신이 암호화되지 않아 보안 문제가 발생할 수 있으며, 특히 Let's Encrypt의 무료 인증서를 사용해 도메인 대신 IP 기반으로 SSL을 설정하려고 했습니다. Let's Encrypt의 HTTP-01 인증 방식이 IP 기반 인증을 지원하지 않으므로 SSL 발급에 실패했습니다. 대안으로 openssl을 사용해 자체 서명 인증서를 발급하고, 내부 네트워크 통신에 적용하는 방법을 선택할 수 있었습니다.
    상황이 이런데도 한반도 정세를 바로잡고 중심에 서야 할 윤석열 정부는 김건희 여사 관련 의혹과 명태균씨의 각종 폭로 등으로 휘청이고 있고, 제1야당인 더불어민주당은 정권 퇴진을 위한 대규모 주말(2일) 집회를 예고했다.

1일 경기일보 취재를 종합하면 5일 앞으로 다가온 미국 대선에서 도널드 트럼프가 당선할 경우 한반도 정책에 큰 변화를 불러올 전망이다. 이 경우 취임 후 ‘한·미·일 공조’에 치중한 윤석열 정부의 대북 강경책은 큰 위기에 직면할 수 있다.

반면 민주당 해리스 미국 부통령이 당선할 경우 조 바이든 대통령과 끈끈한 공조를 다졌던 윤 정부의 대북 기조는 상당 기간 유지될 것으로 보인다.

야생 사과는 중앙아시아와 중국 대륙 사이에 위치한 톈산 산맥과 타림 분지가 원산지로, 이후 전 세계에 퍼지게 되었다. 카자흐스탄의 최대도시 알마티나 신장 위구르 자치구의 도시 알말리크 같은 톈산 산맥 인근 도시 이름들의 어원이 사과이다.
    """

    # max_chunk_size는 문장을 분리한 이후 각 청크의 최대 길이. 즉, src_text를 문장 단위로 분리한 후, 각 청크가 max_chunk_size보다 길어지지 않도록 제한하는 역할
    max_chunk_size = 100
    
    # 텍스트를 문장으로 분리
    sentences = src_text.split(". ")
    
    # 문장별로 인코딩하고 의미 기반 임베딩을 추출
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(inputs["input_ids"])[0]
            sentence_embedding = encoder_outputs.mean(dim=1).squeeze().cpu().numpy()
            sentence_embeddings.append(sentence_embedding)
    
    # 의미 기반 청킹 로직: 문장 간 코사인 유사도 계산
    chunks = []
    current_chunk = sentences[0]
    current_chunk_embedding = sentence_embeddings[0]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity([current_chunk_embedding], [sentence_embeddings[i]])[0][0]
        
        # 유사도가 임계값 이하로 떨어지면 청크 분리
        if similarity < 0.75 or len(current_chunk) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i]
            current_chunk_embedding = sentence_embeddings[i]
        else:
            current_chunk += ". " + sentences[i]
            current_chunk_embedding = (current_chunk_embedding + sentence_embeddings[i]) / 2  # 평균 임베딩 업데이트

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print("chunks", chunks)
    return {"chunks": chunks}

# {
#   "chunks": [
#     "Anytalk 프로젝트에서 로컬 네트워크와 외부 통신을 보안하기 위해 IP 기반 SSL 인증서를 설정할 필요가 있었습니다. SSL 인증서가 없으면 외부와의 통신이 암호화되지 않아 보안 문제가 발생할 수 있으며, 특히 Let's Encrypt의 무료 인증서를 사용해 도메인 대신 IP 기반으로 SSL을 설정하려고 했습니다",
#     "Let's Encrypt의 HTTP-01 인증 방식이 IP 기반 인증을 지원하지 않으므로 SSL 발급에 실패했습니다. 대안으로 openssl을 사용해 자체 서명 인증서를 발급하고, 내부 네트워크 통신에 적용하는 방법을 선택할 수 있었습니다.\n    상황이 이런데도 한반도 정세를 바로잡고 중심에 서야 할 윤석열 정부는 김건희 여사 관련 의혹과 명태균씨의 각종 폭로 등으로 휘청이고 있고, 제1야당인 더불어민주당은 정권 퇴진을 위한 대규모 주말(2일) 집회를 예고했다.\n\n1일 경기일보 취재를 종합하면 5일 앞으로 다가온 미국 대선에서 도널드 트럼프가 당선할 경우 한반도 정책에 큰 변화를 불러올 전망이다",
#     "이 경우 취임 후 ‘한·미·일 공조’에 치중한 윤석열 정부의 대북 강경책은 큰 위기에 직면할 수 있다.\n\n반면 민주당 해리스 미국 부통령이 당선할 경우 조 바이든 대통령과 끈끈한 공조를 다졌던 윤 정부의 대북 기조는 상당 기간 유지될 것으로 보인다.\n\n야생 사과는 중앙아시아와 중국 대륙 사이에 위치한 톈산 산맥과 타림 분지가 원산지로, 이후 전 세계에 퍼지게 되었다",
#     "카자흐스탄의 최대도시 알마티나 신장 위구르 자치구의 도시 알말리크 같은 톈산 산맥 인근 도시 이름들의 어원이 사과이다."
#   ]
# }