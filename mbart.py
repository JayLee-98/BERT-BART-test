
from fastapi import FastAPI, Request
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

app = FastAPI()

# mBART-50 모델과 토크나이저 불러오기
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

@app.post("/mbartSummarize")
async def summarize(src_text: Request):
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
    # 요약 작업 수행
    tokenizer.src_lang = "ko_KR"  # 소스 언어 코드 설정
    model_inputs = tokenizer(src_text, return_tensors="pt")

    # 요약 생성
    summary_ids = model.generate(
        model_inputs['input_ids'],
        max_length=200,  # 최대 요약 길이를 늘림
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary)  # 요약된 한국어 결과 출력