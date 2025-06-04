import pickle, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainingArguments, TrainerCallback
import random, time, numpy as np, nltk
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pad_sequence

# 1. 데이터 로딩 (labels를 gt_response로)
def load_story_data(filepath):
    with open(filepath, "rb") as f:
        utterance, response, narrative, gt_response, y_true, genre = pickle.load(f)
    data = []
    for i in range(len(utterance)):
        example = {
            "idx": i,
            "utterance": utterance[i],
            "narrative": narrative[i],
            "genre": genre[i] if len(genre[i]) > 0 else [10],  # 빈 장르 패딩
            "gt_response": gt_response[i],  # <--- 학습 정답
        }
        data.append(example)
    return data

# 2. Dataset (labels == gt_response)
class StoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "utterance": torch.LongTensor(sample["utterance"]),
            "narrative": torch.LongTensor(sample["narrative"]),
            "genre": torch.LongTensor(sample["genre"]),
            "labels": torch.LongTensor(sample["gt_response"])  # <--- 라벨은 gt_response
        }

# 3. 레이어
class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8): super().__init__(); self.layernorm = nn.LayerNorm(features, eps=epsilon)
    def forward(self, x): return self.layernorm(x)

class multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads=4, dropout_rate=0):
        super().__init__()
        self.num_units, self.num_heads = num_units, num_heads
        self.Q_proj, self.K_proj, self.V_proj = nn.Linear(num_units, num_units), nn.Linear(num_units, num_units), nn.Linear(num_units, num_units)
        self.output_dropout, self.norm = nn.Dropout(p=dropout_rate), layer_normalization(num_units)
    def forward(self, queries, keys, values, mask=None):
        Q, K, V = self.Q_proj(queries), self.K_proj(keys), self.V_proj(values)
        B, tgt_len, D = Q.shape; _, src_len, _ = K.shape
        Q, K, V = [x.view(B, -1, self.num_heads, D//self.num_heads).transpose(1,2) for x in (Q, K, V)]
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (D//self.num_heads)**0.5
        if mask is not None: scores = scores.masked_fill(mask==0, -1e9)
        attn = self.output_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, tgt_len, D)
        return self.norm(context + queries)

class feedforward(nn.Module):
    def __init__(self, in_channels, num_units=[256,256]):
        super().__init__()
        self.fc1, self.fc2, self.norm = nn.Linear(in_channels,num_units[0]), nn.Linear(num_units[0],num_units[1]), layer_normalization(in_channels)
    def forward(self, x): h = F.relu(self.fc1(x)); h = self.fc2(h); return self.norm(h + x)

# 4. 모델
class Seq2SeqScriptWriter(nn.Module):
    def __init__(self, embedding_matrix, num_genres, hidden_dim=256, num_layers=1, num_heads=4, max_len=50, pad_idx=19451, genre_pad_idx=10):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=pad_idx)
        emb_dim = embedding_matrix.shape[1]
        self.genre_emb = nn.Embedding(num_genres, emb_dim, padding_idx=genre_pad_idx)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.enc_ln, self.dec_ln = layer_normalization(hidden_dim), layer_normalization(hidden_dim)
        self.mha = multihead_attention(num_units=hidden_dim, num_heads=num_heads)
        self.ffn = feedforward(hidden_dim, num_units=[hidden_dim*2, hidden_dim])
        self.out = nn.Linear(hidden_dim, embedding_matrix.shape[0])
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.genre_pad_idx = genre_pad_idx

    def forward(self, narrative, utterance, genre, labels=None, teacher_forcing_ratio=0.5):
        nar_emb = self.embedding(narrative)
        utt_emb = self.embedding(utterance)
        if utt_emb.dim() == 4:
            B, n_utt, seq_len, D = utt_emb.shape
            utt_emb = utt_emb.view(B, n_utt * seq_len, D)
        # 장르 임베딩 평균
        genre_mask = (genre != self.genre_pad_idx).float().unsqueeze(-1)
        genre_emb_all = self.genre_emb(genre)
        genre_emb_sum = (genre_emb_all * genre_mask).sum(dim=1)
        genre_cnt = genre_mask.sum(dim=1).clamp(min=1)
        genre_emb = (genre_emb_sum / genre_cnt).unsqueeze(1)

        enc_input = torch.cat([genre_emb, nar_emb, utt_emb], dim=1)  # (B, L, D)
        enc_out, (h, c) = self.encoder(enc_input)
        enc_out = self.enc_ln(enc_out)
        batch_size = narrative.size(0)
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            target_len = labels.size(1)
        else:
            target_len = self.max_len
        input_token = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=narrative.device)
        outputs, dec_hidden, dec_cell = [], h, c
        for t in range(target_len):
            input_emb = self.embedding(input_token)
            dec_out, (dec_hidden, dec_cell) = self.decoder(input_emb, (dec_hidden, dec_cell))
            dec_out_ln = self.dec_ln(dec_out)
            attn_context = self.mha(dec_out_ln, enc_out, enc_out)
            dec_out_ffn = self.ffn(dec_out_ln + attn_context)
            logits = self.out(dec_out_ffn.squeeze(1))
            outputs.append(logits.unsqueeze(1))
            if labels is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = labels[:, t].unsqueeze(1)
            else:
                input_token = logits.argmax(dim=1, keepdim=True)
        outputs = torch.cat(outputs, dim=1)
        if labels is not None:
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_idx
            )
            return {"loss": loss, "logits": outputs}
        else:
            return {"logits": outputs}

# 5. Collate function
def collate_fn(batch, genre_pad_idx=10, text_pad_idx=19451):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "genre":
            max_genre_len = max(len(item[k]) for item in batch)
            genre_padded = []
            for item in batch:
                genre_list = item[k]
                if isinstance(genre_list, torch.Tensor):
                    genre_list = genre_list.tolist()
                padded = genre_list + [genre_pad_idx] * (max_genre_len - len(genre_list))
                genre_padded.append(torch.LongTensor(padded))
            out[k] = torch.stack(genre_padded, dim=0)
        elif k == "labels":
            out[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=text_pad_idx)
        else:
            out[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=text_pad_idx)
    return out

# 6. Subset
def random_subset(dataset, n, seed=42):
    random.seed(seed); indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)

# 7. Vocab, Genre 매핑
def load_vocab(vocab_path):
    idx2word, word2idx = {}, {}
    with open(vocab_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            idx2word[idx], word2idx[word] = word, idx
    return idx2word, word2idx
# idx2word, word2idx = load_vocab('./data/vocab_final.txt')  # <-- import용은 주석
genre2idx = {'SF': 0, '공포(호러)': 1, '드라마': 2, '멜로/로맨스': 3, '미스터리': 4, '스릴러': 5, '액션': 6, '전쟁': 7, '코미디': 8, '판타지': 9}
idx2genre = {v: k for k, v in genre2idx.items()}

# 8. Decode & Metric
def decode_tokens(token_ids, idx2word, sep_idx=28, unk_idx=1490, pad_idx=19451):
    decoded = []
    for seq in token_ids:
        tokens = []
        for t in seq:
            t = int(t)
            if t == sep_idx or t == unk_idx or t == pad_idx:
                continue
            tokens.append(idx2word.get(t, '[UNK]'))
        text = ""
        for tok in tokens:
            if tok.startswith("##") and len(text) > 0:
                text += tok[2:]
            else:
                if len(text) > 0: text += " "
                text += tok
        decoded.append(text)
    return decoded

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print("logits shape:", logits.shape, "labels shape:", labels.shape)
    pred_ids = np.argmax(logits, axis=-1)
    pred_texts, label_texts = decode_tokens(pred_ids, idx2word), decode_tokens(labels, idx2word)
    bleu_scores = [
        nltk.translate.bleu_score.sentence_bleu(
            [label.split()], pred.split(),
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
        )
        for pred, label in zip(pred_texts, label_texts)
    ]
    avg_bleu = np.mean(bleu_scores)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_ls = [scorer.score(label, pred)['rougeL'].fmeasure for pred, label in zip(pred_texts, label_texts)]
    avg_rougeL = np.mean(rouge_ls)
    print("BLEU:", avg_bleu, "ROUGE-L:", avg_rougeL)
    return {"bleu": avg_bleu, "rougeL": avg_rougeL}

# 9. 콜백
class BatchMetricLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"[Batch {state.global_step}] ", end='')
            for k, v in logs.items():
                if "loss" in k or "accuracy" in k:
                    print(f"{k}: {v:.4f} ", end='')
            print()
class EpochTimeLogger(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        state.epoch_start_time = time.time()
    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - getattr(state, "epoch_start_time", time.time())
        print(f"\n[Epoch {int(state.epoch)} finished] Time taken: {elapsed/60:.2f}분 ({elapsed:.2f}초)\n")

# ======= 아래는 직접 실행할 때만 동작 ===========
if __name__ == "__main__":
    # 데이터 및 임베딩 불러오기
    train_data = load_story_data('./data/train_final.pkl')
    dev_data = load_story_data('./data/dev_final.pkl')
    test_data = load_story_data('./data/test_final.pkl')
    embedding_matrix = pickle.load(open('./data/embeddings_final.pkl', 'rb'))
    print("파일 불러오기 완료")

    train_dataset, dev_dataset, test_dataset = map(StoryDataset, [train_data, dev_data, test_data])
    print("데이터셋 생성 완료")

    train_subset = random_subset(train_dataset, min(140000, len(train_dataset)))
    dev_subset = random_subset(dev_dataset, min(10000, len(dev_dataset)))
    test_subset = random_subset(test_dataset, min(10000, len(test_dataset)))
    print("서브셋 생성 완료")

    idx2word, word2idx = load_vocab('./data/vocab_final.txt')

    num_genres = 11  # 0~9 실제장르 + 10(genre_pad_idx)
    model = Seq2SeqScriptWriter(embedding_matrix, num_genres=num_genres, pad_idx=19451, genre_pad_idx=10)
    training_args = TrainingArguments(
        output_dir='./results', per_device_train_batch_size=100, per_device_eval_batch_size=64,
        num_train_epochs=3, save_strategy="epoch",
        logging_steps=50, report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=dev_subset,
        data_collator=lambda batch: collate_fn(batch, genre_pad_idx=10, text_pad_idx=19451),
        compute_metrics=compute_metrics,
        callbacks=[BatchMetricLogger, EpochTimeLogger]
    )

    print("학습 시작")
    trainer.train()
    print("테스트 시작")
    metrics = trainer.evaluate(test_subset)
    print("Testset metrics:", metrics)
    pred_out = trainer.predict(test_subset)
    print("Testset metrics:", pred_out.metrics)
