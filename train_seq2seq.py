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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"(\w)'(\w)", r"\1' \2", text) 
    return re.findall(r"\w+|[^\w\s]", text)

# seed
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

# -------------------- Data Loading --------------------
N_SAMPLES = 2000000

def detect_delimiter(file_path):
    with open(file_path, 'rb') as f:
        first_line = f.readline().decode('utf-8', errors='ignore').strip()
        if '\t' in first_line:
            return '\t', 'utf-8'
        elif ',' in first_line:
            return ',', 'utf-8'
        else:
            # Try other encodings
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
    raise ValueError("Cannot detect delimiter. Please check file format.")
print(f"Detected delimiter: {repr(delimiter)}, encoding: {encoding}")

data_rows = []
with open('en-fr.csv', 'r', encoding=encoding) as f:
    reader = csv.reader(f, delimiter=delimiter, quotechar='"')
    # Skip possible header row
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

# -------------------- Data Preprocessing --------------------
# Length statistics
df['en_len'] = df['en'].str.len()
df['fr_len'] = df['fr'].str.len()
print("\nText length statistics (characters):")
print("English - mean: {:.1f}, min: {}, max: {}".format(df['en_len'].mean(), df['en_len'].min(), df['en_len'].max()))
print("French  - mean: {:.1f}, min: {}, max: {}".format(df['fr_len'].mean(), df['fr_len'].min(), df['fr_len'].max()))

# Cleaning: lowercase, strip spaces, filter by token length
df['en'] = df['en'].str.lower().str.strip()
df['fr'] = df['fr'].str.lower().str.strip()
df = df[
    (df['en'].apply(lambda x: len(tokenize(x)) >= 1)) &
    (df['en'].apply(lambda x: len(tokenize(x)) <= 128)) &
    (df['fr'].apply(lambda x: len(tokenize(x)) >= 1)) &
    (df['fr'].apply(lambda x: len(tokenize(x)) <= 128))
].reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")

# Remove temporary length columns
df = df.drop(columns=['en_len', 'fr_len'])

# Split train/test set
train_df, temp_df = train_test_split(df, test_size=0.1, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Build vocabulary
def build_vocab(texts, min_freq=2, max_size=20000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
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

# Dataset class
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
# -------------------- Seq2Seq Model Definition --------------------
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features

        hidden, cell = self.encoder(src, src_len)
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(device)
        input_token = tgt[:, 0].unsqueeze(1)
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

# -------------------- Training and Evaluation Functions --------------------
def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for src, tgt, src_len, _ in tqdm(dataloader, desc='Training'):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_len, tgt, teacher_forcing_ratio)
        loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_len, _ in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, src_len, tgt, teacher_forcing_ratio=0)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def translate(model, src_text, src_vocab, tgt_vocab, max_len=50, beam_size=5):
    model.eval()

    tokens = [src_vocab.get(w, unk_idx) for w in tokenize(src_text)]
    src_tensor = torch.tensor([bos_idx] + tokens + [eos_idx], dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([src_tensor.size(1)]).to(device)

    hidden, cell = model.encoder(src_tensor, src_len)

    # beam: (sequence, score, hidden, cell)
    beams = [([bos_idx], 0.0, hidden, cell)]

    for _ in range(max_len):
        new_beams = []

        for seq, score, h, c in beams:
            last_token = seq[-1]

            if last_token == eos_idx:
                new_beams.append((seq, score, h, c))
                continue

            tgt_input = torch.tensor([[last_token]], dtype=torch.long).to(device)

            output, h_new, c_new = model.decoder(tgt_input, h, c)

            log_probs = torch.log_softmax(output.squeeze(1), dim=-1).squeeze(0)

            # temperature
            log_probs = log_probs / 0.8

            if len(seq) > 2:
                log_probs[seq[-1]] -= 1.0


            if len(seq) > 2:
                log_probs[seq[-2]] -= 0.5

            if len(seq) > 3:
                log_probs[seq[-3]] -= 0.2

            for tok in set(seq[-5:]):
                log_probs[tok] -= 0.5

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = topk_indices[i].item()
                new_score = score + topk_log_probs[i].item()

                new_seq = seq + [next_token]

                new_beams.append((new_seq, new_score, h_new.clone(), c_new.clone()))

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** 0.7),
            reverse=True
        )[:beam_size]

        if all(seq[-1] == eos_idx for seq, _, _, _ in beams):
            break

    best_seq = beams[0][0]

    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

    translation = [
        inv_tgt_vocab.get(tok, '<unk>')
        for tok in best_seq[1:-1]
        if tok not in (bos_idx, eos_idx, pad_idx)
    ]

    return ' '.join(translation)

def compute_bleu(model, dataloader, src_vocab, tgt_vocab):
    model.eval()
    references = []
    hypotheses = []
    for src, tgt, _, _ in tqdm(dataloader, desc='BLEU'):
        src = src.to(device)
        for i in range(src.size(0)):
            src_text_tokens = src[i].tolist()
            inv_src_vocab = {v: k for k, v in src_vocab.items()}
            src_text = ' '.join([inv_src_vocab.get(tok, '<unk>') for tok in src_text_tokens if tok not in (bos_idx, eos_idx, pad_idx)])
            ref_tokens = tgt[i].tolist()
            inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
            ref = [inv_tgt_vocab.get(tok, '<unk>') for tok in ref_tokens if tok not in (bos_idx, eos_idx, pad_idx)]
            hyp = translate(model, src_text, src_vocab, tgt_vocab).split()
            if not hyp:
                hyp = ['<empty>']
            references.append([ref])
            hypotheses.append(hyp)
    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu

# -------------------- Main Training Procedure --------------------
os.makedirs('result', exist_ok=True)

embed_dim = 256
hidden_dim = 512
num_layers = 2
dropout = 0.3
encoder = EncoderLSTM(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
decoder = DecoderLSTM(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
num_epochs = 400
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

for epoch in range(1, num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs}")

    tf_ratio = max(0.6 - epoch * 0.02, 0.2)

    train_loss = train_epoch(model, train_loader, optimizer, criterion, teacher_forcing_ratio=tf_ratio)
    val_loss = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'result/Seq2Seq_best.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

model.load_state_dict(torch.load('result/Seq2Seq_best.pth'))
test_loss = evaluate(model, test_loader, criterion)
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

bleu = compute_bleu(model, test_loader, src_vocab, tgt_vocab)
print(f"Seq2Seq BLEU Score: {bleu:.4f}")

# Plot loss curve
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Seq2Seq Training Curve')
plt.legend()
plt.grid(True)
plt.savefig('result/Seq2Seq_loss.png')
plt.close()

# Translation examples
sample_indices = [0, 1, 2, 3, 4]
print("\n===== Translation Examples =====")
with open('result/seq2seq_translations.txt', 'w', encoding='utf-8') as f:
    f.write("Seq2Seq Translation Examples\n\n")
    for idx in sample_indices:
        src_sample = test_df['en'].iloc[idx]
        ref_sample = test_df['fr'].iloc[idx]
        hyp = translate(model, src_sample, src_vocab, tgt_vocab)
        print(f"Source: {src_sample}")
        print(f"Reference: {ref_sample}")
        print(f"Seq2Seq: {hyp}\n")
        f.write(f"Source: {src_sample}\nReference: {ref_sample}\nSeq2Seq: {hyp}\n\n")
print("Results saved to result/Seq2Seq_loss.png and result/seq2seq_translations.txt")