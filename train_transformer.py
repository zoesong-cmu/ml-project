import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
from sklearn.model_selection import train_test_split
import math
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')


import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"(\w)'(\w)", r"\1' \2", text) 
    return re.findall(r"\w+|[^\w\s]", text)

def detokenize(tokens):
    sentence = ' '.join(tokens)
    sentence = sentence.replace(" ' ", "'")
    return sentence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

N_SAMPLES = 2000000

def detect_delimiter(file_path):
    with open(file_path, 'rb') as f:
        first_line = f.readline().decode('utf-8', errors='ignore').strip()
        if '\t' in first_line:
            return '\t', 'utf-8'
        elif ',' in first_line:
            return ',', 'utf-8'
        else:
            for enc in ['latin-1', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=enc) as f2:
                        first = f2.readline().strip()
                        if '\t' in first:
                            return '\t', enc
                        elif ',' in first:
                            return ',', enc
                except:
                    continue
    return None, None

delimiter, encoding = detect_delimiter('en-fr.csv')
if delimiter is None:
    raise ValueError("Cannot detect delimiter.")
print(f"Detected delimiter: {repr(delimiter)}, encoding: {encoding}")

data_rows = []
with open('en-fr.csv', 'r', encoding=encoding) as f:
    reader = csv.reader(f, delimiter=delimiter, quotechar='"')
    first_row = next(reader, None)
    if first_row and len(first_row) >= 2:
        if 'en' in first_row[0].lower() and 'fr' in first_row[1].lower():
            print("Skipping header row")
        else:
            en, fr = first_row[0].strip(), first_row[1].strip()
            if en and fr:
                data_rows.append((en, fr))
    for i, row in enumerate(reader):
        if len(data_rows) >= N_SAMPLES:
            break
        if len(row) < 2:
            continue
        en, fr = row[0].strip(), row[1].strip()
        if en and fr:
            data_rows.append((en, fr))

print(f"Loaded {len(data_rows)} valid rows")
if len(data_rows) == 0:
    raise ValueError("No valid data loaded.")
df = pd.DataFrame(data_rows, columns=['en', 'fr'])

df['en_len'] = df['en'].str.len()
df['fr_len'] = df['fr'].str.len()
print("\nText length statistics (characters):")
print("English - mean: {:.1f}, min: {}, max: {}".format(df['en_len'].mean(), df['en_len'].min(), df['en_len'].max()))
print("French  - mean: {:.1f}, min: {}, max: {}".format(df['fr_len'].mean(), df['fr_len'].min(), df['fr_len'].max()))

df['en'] = df['en'].str.lower().str.strip()
df['fr'] = df['fr'].str.lower().str.strip()
df = df[
    (df['en'].apply(lambda x: len(tokenize(x))) >= 1) &
    (df['en'].apply(lambda x: len(tokenize(x))) <= 128) &
    (df['fr'].apply(lambda x: len(tokenize(x))) >= 1) &
    (df['fr'].apply(lambda x: len(tokenize(x))) <= 128)
].reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")
df = df.drop(columns=['en_len', 'fr_len'])

train_df, temp_df = train_test_split(df, test_size=0.1, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

def build_vocab(texts, min_freq=2, max_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq and len(vocab) < max_size:
            vocab[word] = len(vocab)
    return vocab

src_vocab = build_vocab(train_df['en'].tolist())
tgt_vocab = build_vocab(train_df['fr'].tolist())
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
print(f"Source vocab size: {src_vocab_size}")
print(f"Target vocab size: {tgt_vocab_size}")

pad_idx = src_vocab['<pad>']
unk_idx = src_vocab['<unk>']
bos_idx = tgt_vocab['<bos>']
eos_idx = tgt_vocab['<eos>']

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]
        src_indices = [self.src_vocab['<bos>']] + [self.src_vocab.get(w, unk_idx) for w in tokenize(src)] + [self.src_vocab['<eos>']]
        tgt_indices = [self.tgt_vocab['<bos>']] + [self.tgt_vocab.get(w, unk_idx) for w in tokenize(tgt)] + [self.tgt_vocab['<eos>']]
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded, torch.tensor(src_lens), torch.tensor(tgt_lens)

train_dataset = TranslationDataset(train_df['en'].tolist(), train_df['fr'].tolist(), src_vocab, tgt_vocab)
val_dataset   = TranslationDataset(val_df['en'].tolist(), val_df['fr'].tolist(), src_vocab, tgt_vocab)
test_dataset  = TranslationDataset(test_df['en'].tolist(), test_df['fr'].tolist(), src_vocab, tgt_vocab)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x):
    # x: (batch, nhead, seq_len, d_k)
    batch, nhead, seq_len, d_k = x.size()
    assert d_k % 2 == 0, "RoPE requires even head dimension."

    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)  # (seq_len,)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32) / d_k))
    freqs = torch.einsum("i,j->ij", pos, inv_freq)  # (seq_len, d_k/2)

    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_k)
    cos = emb.cos()[None, None, :, :]        # (1, 1, seq_len, d_k)
    sin = emb.sin()[None, None, :, :]        # (1, 1, seq_len, d_k)

    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        seq_len_q, batch_size, _ = query.size()
        seq_len_k, _, _ = key.size()

        Q = self.w_q(query)  # (seq_len_q, batch, d_model)
        K = self.w_k(key)    # (seq_len_k, batch, d_model)
        V = self.w_v(value)  # (seq_len_k, batch, d_model)

        Q = Q.permute(1, 0, 2).contiguous().view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = K.permute(1, 0, 2).contiguous().view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = V.permute(1, 0, 2).contiguous().view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)

        Q = apply_rope(Q)
        K = apply_rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask == 0, -1e9)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            scores = scores + attn_mask  

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = output.permute(1, 0, 2)
        output = self.w_o(output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_attn_mask=None):
        attn_output = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        src = src.transpose(0, 1)  # (src_len, batch)
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        for layer in self.layers:
            src_emb = layer(src_emb, key_padding_mask=src_key_padding_mask)
        return src_emb

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_attn_mask=None):
        tgt = tgt.transpose(0, 1)  # (tgt_len, batch)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        for layer in self.layers:
            tgt_emb = layer(tgt_emb, enc_output, src_key_padding_mask, tgt_key_padding_mask, tgt_attn_mask)
        output = self.fc_out(tgt_emb)  # (tgt_len, batch, vocab_size)
        return output.transpose(0, 1)  # (batch, tgt_len, vocab_size)

class TransformerCustom(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_layers=2, d_ff=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, nhead, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, nhead, d_ff, num_layers, max_len, dropout)

    def generate_padding_mask(self, seq, pad_idx):
        return (seq != pad_idx).bool()

    def generate_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_key_padding_mask = self.generate_padding_mask(src, pad_idx)
        tgt_key_padding_mask = self.generate_padding_mask(tgt, pad_idx)
        tgt_len = tgt.size(1)
        tgt_attn_mask = self.generate_subsequent_mask(tgt_len).to(device)
        enc_output = self.encoder(src, src_key_padding_mask)
        output = self.decoder(tgt, enc_output, src_key_padding_mask, tgt_key_padding_mask, tgt_attn_mask)
        return output

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt, _, _ in tqdm(dataloader, desc='Training'):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, _, _ in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def translate(model, sentence, src_vocab, tgt_vocab, device, beam_size=5, max_len=50):
    model.eval()

    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}

    tokens = tokenize(sentence)
    src = torch.tensor([[
        src_vocab['<bos>']
    ] + [
        src_vocab.get(tok, src_vocab['<unk>']) for tok in tokens
    ] + [
        src_vocab['<eos>']
    ]], device=device)
    
    src_mask = model.generate_padding_mask(src, pad_idx)
    memory = model.encoder(src, src_mask)
    
    sequences = [([tgt_vocab['<bos>']], 0.0)]
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score in sequences:
            if seq[-1] == tgt_vocab['<eos>']:
                all_candidates.append((seq, score))
                continue
                
            tgt = torch.tensor([seq], device=device)

            tgt_len = tgt.size(1)
            tgt_mask = model.generate_subsequent_mask(tgt_len).to(device)
            output = model.decoder(tgt, memory, tgt_attn_mask=tgt_mask)

            temperature = 0.8
            log_probs = F.log_softmax(output[:, -1, :] / temperature, dim=-1)
            
            for tok in set(seq[-5:]):
                log_probs[0, tok] -= 0.4

            topk = torch.topk(log_probs, beam_size)
            
            for i in range(beam_size):
                token = topk.indices[0, i].item()
                candidate = seq + [token]
                candidate_score = score + topk.values[0, i].item()
                all_candidates.append((candidate, candidate_score))
        
        sequences = sorted(
            all_candidates,
            key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** 0.7),
            reverse=True
        )[:beam_size]
    
    best_seq = sequences[0][0]
    return [
        tgt_vocab_inv[idx] 
        for idx in best_seq 
        if idx not in (tgt_vocab['<bos>'], tgt_vocab['<eos>'])
    ]


def compute_bleu(model, dataloader, src_vocab, tgt_vocab):
    model.eval()
    references = []
    hypotheses = []

    idx_counter = 0

    for src, tgt, _, _ in tqdm(dataloader, desc='BLEU'):
        src = src.to(device)
        for i in range(src.size(0)):

            src_text = test_df['en'].iloc[idx_counter]
            idx_counter += 1                             

            ref_tokens = tgt[i].tolist()
            inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

            ref = [inv_tgt_vocab.get(tok, '<unk>') 
                   for tok in ref_tokens 
                   if tok not in (bos_idx, eos_idx, pad_idx)]

            hyp = translate(model, src_text, src_vocab, tgt_vocab, device)

            if not hyp:
                hyp = ['<empty>']

            references.append([ref])
            hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu

os.makedirs('result_Transformer', exist_ok=True)

d_model = 256
nhead = 4
num_layers = 2
d_ff = 512
dropout = 0.1
model = TransformerCustom(src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, d_ff, 5000, dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
num_epochs = 400
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

for epoch in range(1, num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'result_Transformer/Transformer_best.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

model.load_state_dict(torch.load('result_Transformer/Transformer_best.pth'))
test_loss = evaluate(model, test_loader, criterion)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

bleu = compute_bleu(model, test_loader, src_vocab, tgt_vocab)
print(f"Transformer BLEU Score: {bleu:.4f}")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer Training Curve')
plt.legend()
plt.grid(True)
plt.savefig('result_Transformer/Transformer_loss.png')
plt.close()

sample_indices = [0, 1, 2, 3, 4]
print("\n===== Final Translation Examples =====")
with open('result_Transformer/transformer_translations.txt', 'w', encoding='utf-8') as f:
    f.write("Transformer Translation Examples\n\n")
    for idx in sample_indices:
        src_sample = test_df['en'].iloc[idx]
        ref_sample = test_df['fr'].iloc[idx]
        hyp = translate(model, src_sample, src_vocab, tgt_vocab, device)
        print(f"Source: {src_sample}")
        print(f"Reference: {ref_sample}")
        print(f"Transformer: {detokenize(hyp)}\n")
        f.write(f"Source: {src_sample}\nReference: {ref_sample}\nTransformer: {detokenize(hyp)}\n\n")
print("Results saved to result_Transformer/Transformer_loss.png and result_Transformer/transformer_translations.txt")