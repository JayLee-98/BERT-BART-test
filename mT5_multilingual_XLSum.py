# supports multi-lang summarization

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""
# article_text = """이런 가운데 이날 국회 운영위원회에서 대통령실 관련 의혹을 놓고 격론을 벌이면서 야권은 2일 서울 광화문 집회 총동원령에 나섰다.

# 민주당 이재명 대표는 이날 페이스북을 통해 “11월2일 국민행동의 날, 주권자의 열망을 가득 모아달라”며 “‘악이 승리하는 유일한 조건은 선한 사람들이 아무것도 하지 않는 것’이라 했다”고 말했다.

# 이어 “무너진 희망을 다시 세울 힘도, 새로운 길을 열어젖힐 힘도 ‘행동하는 주권자’에게 있다고 믿는다”며 “다가오는 11월 2일 국민행동의 날, 정의의 파란 물결로 서울역을 뒤덮어 달라”고 요청했다.

# 이 대표는 단문 메시지를 통해서도 “로시난테를 타고 풍차를 향해 돌진하는 돈키호테처럼, 미친 듯이 전쟁을 향해 질주하는 이유는 뭘까”라며 윤석열 정부의 대북 정책을 강하게 비판했다.

# 이와 관련해 여권의 한 핵심 관계자는 이날 경기일보와 통화에서 “대통령실과 여당이 잘못하고 있는 것을 부정하지 않는다”며 “그러나 지금은 미국과 일본의 정권교체 여부와 북·러 군사동맹 등으로 안보위기가 심각한 상황”이라고 전제했다.

# 그러면서 “여야 모두가 자중하고 협력해야 함에도 조국당의 윤 탄핵 집회에 이어 제1야당이 자당 대표 방탄 목적으로 국회 파행에 이어 장외집회까지 예고했다”며 “지금 대한민국은 사실상 내전 상태가 아니냐”고 반문했다."""

article_text = """마토스는 브루노 마스를 닮은 외모와 목소리로 국내외 누리꾼들로부터 큰 호응을 얻고 있으며, 해당 영상은 그가 로제와 브루노 마스의 인기 곡 '아파트(APT.)'를 부르는 모습을 담았다.

마토스는 해당 영상에서도 브루노 마스의 파트를 비슷하게 소화해 많은 환호를 받고 있다.

인스타그램 릴스 영상 속에서는 많은 사람들이 그에게 열광하며 함께 노래를 부르고 그의 이름을 연호하기도 했다. 또한 사진 촬영을 요청하거나 '허그 타임'을 가지는 등 연예인 못지않은 인기를 누리고 있는 모습도 포착됐다.

야구 선수 오타니 쇼헤이와 닮은 외모로 화제가 된 이데구치 요시키도 마찬가지다. 시부야에서 LA다저스 유니폼을 입고 오타니 쇼헤이 광고판 앞에 선 그의 영상은 750만 뷰를 기록하며 큰 인기를 끌었다. 누리꾼들은 '테무산 오타니' '쉬인 오타니' 등 댓글을 달면서 그의 외모가 쇼헤이와 억울하게 닮았다는 반응을 내놓았다.

예명도 중간에 비슷하다는 의미의 '似(사)'자 하나를 더 넣어 '오타니니 쇼헤이(大谷似翔平)'라고 짓고, 지난해부터 본격적으로 '짭(짝퉁) 오타니'로 활동하기 시작했다.

그는 일본 언론과의 인터뷰에서 "예전에는 2000엔(약 1만8000원)짜리 셔츠밖에 못 입었는데, (오타니 닮은꼴로 유명해지면서부터)이제는 4만5000엔(약 40만원)짜리 'BOSS 셔츠'도 살 수 있게 됐다"며 '인생역전'을 실감한다고 전했다.

이런 유명인과 닮은 '페이크 셀럽'이 인기를 끌게 된 것은 어제 오늘만의 일이 아니다. '짭(짝퉁)'→'보급형'→'테무산'으로 변해오면서 꾸준히 화제몰이를 해왔다. 이러한 말들은 실제보다 기대에 미치지 못했거나 다소 애매하게 닮은 상황에 사용한다. 요즘 들어선 틱톡, 인스타그램, 유튜브에서 삽시간에 확산돼 그 파급력이 더 강해지고 있다.

과거 KBS 예능 '무엇이든 물어보살'에서 자신을 '테무산 박서준'으로 소개한 한 남성의 영상도 약 500만 뷰를 기록했다.

같은 방송국 예능 프로그램 '메소드 클럽'에서 개그맨 곽범이 배우 정우성을 흉내 낸 영상도 온라인에서 '테무에서 산 정우성'이라는 제목으로 빠르게 확산되며 큰 주목을 받았다.

이외에도 테무산 장원영, 카리나, 차은우 등 인플루언서들이 유명인의 메이크업, 표정, 말투 등을 따라 하는 영상을 올려 연일 높은 조회수를 기록하고 있다.

닮은꼴로 지목되는 것은 그 유명인의 인기를 보여주는 지표가 되기도 한다. '강남스타일'이 빌보드에 진입했을 당시 싸이 닮은꼴 인플루언서들이 큰 주목을 받으며 관련 영상이 화제가 됐고, 김정은 북한 위원장과 도널드 트럼프 전 미국 대통령의 회담이 세계적 이슈가 되자 두 사람의 닮은꼴이 등장해 언론의 관심을 모았다.

사진과 영상에서는 닮았지만, 실제 방송에서 전혀 다른 모습으로 화제가 된 '닮은꼴 호소인'들도 있다.

송강 닮은꼴로 조회수 1억 뷰를 기록한 유튜버 오재형은 누리꾼들로부터 "훈남이긴 한데 송강은 좀 아닌 듯" "이젠 영상도 못 믿겠네" 등의 반응을 받고 있다. 박보검을 닮았다고 주장했던 민서공이 역시SNS에서 사진과 영상이 화제를 모았지만, 실제 방송에서 나와 전혀 다른 외모만 인증했다는 의견이 많다.

이처럼 '페이크 셀럽'들이 온라인상에서 많은 관심을 모으고 있는 가운데, 자신이 동경하는 연예인과 닮고 싶은 마음이 과도한 집착으로 변하는 경우도 있다.

영국에 사는 한 남성 인플루언서는 방탄소년단 지민을 닮으려고 32번 성형수술을 받았다. 이에 방탄소년단 팬들은 "자기 자신을 있는 그대로 사랑하자는 것이 방탄소년단의 메세지인데 그 의미에 반하는 행동이다"고 말했고, 많은 누리꾼은 "너무 과한 팬심인 것 같다" "전혀 안 닮았다" "제발 그러지 마라" 등 부정적인 의견을 내놓았다."""

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
# YouTube has banned thousands of videos spreading misinformation about Covid vaccines.
# 11월 2일 국민행동의 날을 맞아 정의의 파란 물결로 서울역을 뒤덮어달라고 요청했다.
# 팝스타 로제 마토스가 자신을 닮은 외모로 화제다.