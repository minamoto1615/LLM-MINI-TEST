import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1. 학습 문장 준비
# ---------------------------

sentences = [
    "그는 오늘 잔다",
    "그는 내일 공부한다"
]

# ---------------------------
# 2. 토큰 사전 구축
# ---------------------------

tokens = set(" ".join(sentences).split())
vocab = {t: i for i, t in enumerate(tokens)}
ivocab = {i: t for t, i in vocab.items()}

print("Vocab:", vocab)

# ---------------------------
# 3. 다음 단어 예측용 pair 생성
# ---------------------------

pairs = []
for s in sentences:
    words = s.split()
    for i in range(len(words)-1):
        x = vocab[words[i]]
        y = vocab[words[i+1]]
        pairs.append((x, y))

print("Train pairs:", pairs)

# ---------------------------
# 4. 미니 LLM (초소형 next-token 모델)
# ---------------------------

vocab_size = len(vocab)
embed_dim = 16  # 실제 GPT는 768~4096

model = nn.Sequential(
    nn.Embedding(vocab_size, embed_dim),  # 단어 → 벡터
    nn.Linear(embed_dim, vocab_size)      # 벡터 → 다음 단어 확률
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ---------------------------
# 5. 학습 루프
# ---------------------------

for epoch in range(200):
    total_loss = 0
    for x, y in pairs:
        x_t = torch.tensor([x])  # 입력 토큰
        y_t = torch.tensor([y])  # 정답 토큰

        logits = model(x_t)      # 모델 forward
        loss = loss_fn(logits, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# ---------------------------
# 6. 예측 함수
# ---------------------------

def predict(word):
    x = torch.tensor([vocab[word]])
    logits = model(x)
    prob = torch.softmax(logits, dim=-1)
    idx = torch.argmax(prob).item()
    return ivocab[idx]

# ---------------------------
# 7. 테스트
# ---------------------------

print("\n--- Prediction Test ---")
print("오늘 →", predict("오늘"))     # 잔다
print("내일 →", predict("내일"))     # 공부한다