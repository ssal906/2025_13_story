import torch
import pickle
from seq_model import Seq2SeqScriptWriter

# 1. 데이터 관련 파일 로드
with open('./data/embeddings_final.pkl', 'rb') as f:
    embedding_matrix = pickle.load(f)

def load_vocab(vocab_path):
    idx2word, word2idx = {}, {}
    with open(vocab_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip().split()[0]  # 단어 부분만 추출
            idx2word[idx], word2idx[word] = word, idx
    return idx2word, word2idx


idx2word, word2idx = load_vocab('./data/vocab_final.txt')
genre2idx = {'SF': 0, '공포(호러)': 1, '드라마': 2, '멜로/로맨스': 3, '미스터리': 4, '스릴러': 5, '액션': 6, '전쟁': 7, '코미디': 8, '판타지': 9}

from safetensors.torch import load_file as safe_load

# 모델 인스턴스 생성
num_genres = 11
pad_idx = 19451
genre_pad_idx = 10
model = Seq2SeqScriptWriter(embedding_matrix, num_genres=num_genres, pad_idx=pad_idx, genre_pad_idx=genre_pad_idx)

# safetensors 파일 불러오기
state_dict = safe_load('./model.safetensors')
model.load_state_dict(state_dict)
model.eval()  # 평가모드

# 예시: 입력 텍스트를 토큰 ID로 변환 (word2idx 사용)
utterance_text = "잘 지냈어?"
narrative_text = "나는 오래 전 헤어진 친구를 만났다."
genre_list = ["드라마"]   # 예시

# 5. 실제 입력과 UNK 비율 체크
def text2ids(text, word2idx, unk_idx=1490):
    ids = [word2idx.get(tok, unk_idx) for tok in text.strip().split()]
    unk_count = sum([t == unk_idx for t in ids])
    return ids

utterance_ids = text2ids(utterance_text, word2idx)
narrative_ids = text2ids(narrative_text, word2idx)
genre_ids = [genre2idx.get(g, genre_pad_idx) for g in genre_list]

# 배치(1개)로 텐서 변환
utterance = torch.LongTensor([utterance_ids])
narrative = torch.LongTensor([narrative_ids])
genre = torch.LongTensor([genre_ids])

with torch.no_grad():
    output = model(
        narrative=narrative,
        utterance=utterance,
        genre=genre,
        labels=None  # 추론
    )
    logits = output["logits"]
    pred_ids = logits.argmax(dim=-1).cpu().numpy()

# 1. 예측된 pred_ids 값
print("예측된 pred_ids (앞 10개 row):", pred_ids[:10])

# 2. 디코딩 시 idx2word에 없는 인덱스가 있는지 출력 (디버그)
def decode_tokens_debug(token_ids, idx2word, sep_idx=28, unk_idx=1490, pad_idx=19451):
    decoded = []
    for seq_num, seq in enumerate(token_ids):
        tokens = []
        for pos, t in enumerate(seq):
            t = int(t)
            if t == sep_idx or t == unk_idx or t == pad_idx:
                continue
            word = idx2word.get(t, '[UNK]')
            if word == '[UNK]':
                print(f"[!] idx2word에 없는 토큰 인덱스! seq{seq_num}, pos{pos}, token_id={t}")
            tokens.append(word)
        text = ""
        for tok in tokens:
            if tok.startswith("##") and len(text) > 0:
                text += tok[2:]
            else:
                if len(text) > 0: text += " "
                text += tok
        decoded.append(text)
    return decoded

result_text = decode_tokens_debug(pred_ids, idx2word)
print("모델 생성 문장:", result_text)
