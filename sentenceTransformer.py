# works magic !!! / sentence similarity search
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# 모델 로드
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@app.post("/sentenceSimilarity")
async def sentence_similarity(source_sentence):

    # source_sentence: """대통령실은 1일 더불어민주당이 전날 공개한 윤석열 대통령과 명태균 씨의 통화 녹음 내용에 대해 "윤 대통령은 취임 전후에 공천 개입, 선거 개입과 같은 불법 행위를 한 바가 없다"고 밝혔다."""

    # 기준 문장과 비교할 문장들
    sentences = [
        """이런 가운데 이날 국회 운영위원회에서 대통령실 관련 의혹을 놓고 격론을 벌이면서 야권은 2일 서울 광화문 집회 총동원령에 나섰다. 민주당 이재명 대표는 이날 페이스북을 통해 “11월2일 국민행동의 날, 주권자의 열망을 가득 모아달라”며 “‘악이 승리하는 유일한 조건은 선한 사람들이 아무것도 하지 않는 것’이라 했다”고 말했다. 이어 “무너진 희망을 다시 세울 힘도, 새로운 길을 열어젖힐 힘도 ‘행동하는 주권자’에게 있다고 믿는다”며 “다가오는 11월 2일 국민행동의 날, 정의의 파란 물결로 서울역을 뒤덮어 달라”고 요청했다. 이 대표는 단문 메시지를 통해서도 “로시난테를 타고 풍차를 향해 돌진하는 돈키호테처럼, 미친 듯이 전쟁을 향해 질주하는 이유는 뭘까”라며 윤석열 정부의 대북 정책을 강하게 비판했다. 이와 관련해 여권의 한 핵심 관계자는 이날 경기일보와 통화에서 “대통령실과 여당이 잘못하고 있는 것을 부정하지 않는다”며 “그러나 지금은 미국과 일본의 정권교체 여부와 북·러 군사동맹 등으로 안보위기가 심각한 상황”이라고 전제했다. 그러면서 “여야 모두가 자중하고 협력해야 함에도 조국당의 윤 탄핵 집회에 이어 제1야당이 자당 대표 방탄 목적으로 국회 파행에 이어 장외집회까지 예고했다”며 “지금 대한민국은 사실상 내전 상태가 아니냐”고 반문했다.""",
        """이날 오전 11시께 경찰로부터 "차량이 많이 파손됐다"는 연락을 받고 피해 사실을 알게 된 제보자는 앞 유리가 깨지고, 사이드미러 역시 형체도 알아볼 수 없을 정도로 망가진 상태의 차량을 확인했다. 블랙박스에는 백발의 노인이 나무판자 등의 이용해 차량을 부수는 모습이 고스란히 담겨있었다.""",
        """대한민국 법원(大韓民國 法院, 영어: Court of Korea)은 대한민국에서 포괄적 사법권을 행사하는 일반법원으로, 법원조직법에 따라 조직된다. 헌법 제101조 제2항에 따른 자체적 최고법원인 대법원과 그 하급심 법원들인 각급법원으로 구성되어 있다. 사법권 독립의 보장 사법권의 독립 문서를 참고하십시오. 사법권을 입법권과 행정권으로부터 독립시켜 법관이 재판을 할 때에는 정치적 또는 사회적 압력을 받지 아니하고 다만 헌법과 법률에 의거함으로써 재판의 공정을 기하자는 것이 사법권 독립의 원칙이다. 사법권은 법관으로 구성된 법원에 속하는 것(헌법 제101조 제1항), 법원의 인적 조직에 관한 것(헌법 제102조), 법관은 판결을 함에 있어 헌법과 법률에 의하여 그 양심에 따라 독립하여 심판하는 것(헌법 제103조), 법관의 자격은 법률로써 정하는 것(헌법 제101조 제3항), 법관의 임기는 10년으로 하며 법률이 정하는 바에 의해 연임될 수 있는 것(헌법 제105조 제2항), 법관의 신분 보장을 강화한 것(헌법 제106조), 법관의 인사를 자체적으로 주관하는 것(헌법 제104조) 등의 규정이 바로 그 내용이다."""
    ]

    # 기준 문장을 포함하여 모든 문장의 임베딩 생성
    all_sentences = [source_sentence] + sentences
    embeddings = model.encode(all_sentences)

    # 기준 문장 임베딩과 각 비교 대상 문장의 임베딩 간 코사인 유사도 계산
    source_embedding = embeddings[0]  # 기준 문장의 임베딩
    comparison_embeddings = embeddings[1:]  # 비교 대상 문장들의 임베딩

    # 유사도 계산
    similarities = cosine_similarity([source_embedding], comparison_embeddings)

    # 결과 출력
    for i, sentence in enumerate(sentences):
        print(f"Sentence: '{sentence}'")
        print(f"Cosine Similarity with source sentence: {similarities[0][i]:.4f}\n")

# 일반적인 문장 유사도 계산에서는 코사인 유사도가 주로 사용됩니다. 이유는 코사인 유사도가 고차원 벡터 공간에서 각도를 기반으로 비교하기 때문에, 임베딩 벡터의 크기보다는 방향을 중점으로 비교할 수 있어 유사도 계산에 적합하기 때문입니다.
# 꼭 코사인 유사도만 사용해야 하나요?
# 꼭 코사인 유사도만을 사용할 필요는 없지만, 코사인 유사도가 주로 사용되는 이유는 임베딩 벡터들이 고차원 공간에서 각도를 기반으로 비교될 때, 크기의 차이 없이 비교할 수 있기 때문입니다. 다른 유사도 계산 방법으로는 다음이 있습니다:
# 유클리드 거리 (Euclidean Distance): 두 벡터 간의 직선 거리를 측정합니다. 크기 차이가 결과에 영향을 미칠 수 있습니다.
# 자카드 유사도 (Jaccard Similarity): 이산형 데이터에서 주로 사용되며, 두 집합 간의 공통 요소 비율을 비교합니다.
# 맨해튼 거리 (Manhattan Distance): 두 벡터 간의 축을 따라 이동한 총 거리를 측정합니다.
# 코사인 유사도는 텍스트 임베딩에서 방향성에 초점을 두므로, 문장의 의미적 유사성을 측정하는 데 가장 널리 사용됩니다. 하지만, 특정 목적에 따라 다른 유사도 측정 방법을 사용할 수 있습니다.
# 실제 사용에서는 sentence-transformers와 함께 코사인 유사도를 사용하는 것이 가장 효과적이며, 라이브러리에서 이미 최적화된 유사도 계산 기능을 제공하므로 이를 활용하는 것이 좋습니다.

@app.post("/semanticChunking")
async def semantic_chunking(text: str, threshold: float = 0.8, max_chunk_length: int = 300):
    # 텍스트를 문장으로 분리
    sentences = text.split(". ")
    
    # 문장별 임베딩 생성
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_chunk_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity([current_chunk_embedding], [embeddings[i]])[0][0]
        
        if similarity < threshold or len(" ".join(current_chunk)) > max_chunk_length:
            # 새로운 청크 생성
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_chunk_embedding = embeddings[i]
        else:
            # 현재 청크에 문장 추가 및 평균 임베딩 업데이트
            current_chunk.append(sentences[i])
            current_chunk_embedding = np.mean([current_chunk_embedding, embeddings[i]], axis=0)

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return {"chunks": chunks}