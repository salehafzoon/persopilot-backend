# Improved Persona Extractor Training Code

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertModel, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from torch import nn
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 8  # Reduced for better gradients
EPOCHS = 15     # Increased
TRAIN_SPLIT = 0.8

# Load and preprocess data
data_path = "src/data/ConvAI2/u2t_map_all.json"
with open(data_path, "r") as f:
    raw_data = json.load(f)

all_relations = sorted({ex["triplets"][0]["label"] for ex in raw_data})
relation2id = {rel: i for i, rel in enumerate(all_relations)}
id2relation = {i: rel for rel, i in relation2id.items()}

def convert_to_bio(example):
    triplet = example["triplets"][0]
    tokens = triplet["tokens"]
    head = triplet["head"]
    tail = triplet["tail"]
    relation = triplet["label"]

    labels = ['O'] * len(tokens)
    for idx in head:
        labels[idx] = 'B-SUB'
    if isinstance(tail, list):
        for i, idx in enumerate(tail):
            labels[idx] = 'B-OBJ' if i == 0 else 'I-OBJ'

    return {
        "tokens": tokens,
        "labels": labels,
        "relation_label": relation2id[relation]
    }

bio_data = [convert_to_bio(ex) for ex in raw_data if "triplets" in ex and ex["triplets"]]

# Improved tokenization and alignment
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
label_list = ['O', 'B-SUB', 'B-OBJ', 'I-OBJ']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align(example):
    assert len(example["labels"]) == len(example["tokens"]), "Token-label length mismatch"

    encoding = tokenizer(example["tokens"], is_split_into_words=True, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    word_ids = encoding.word_ids()

    aligned_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:  # First subtoken of a word
            aligned_labels.append(label2id[example["labels"][word_id]])
        else:  # Continuation subtoken
            current_label = example["labels"][word_id]
            if current_label == "B-OBJ":
                aligned_labels.append(label2id["I-OBJ"])  # Convert B-OBJ to I-OBJ for continuation
            elif current_label == "I-OBJ":
                aligned_labels.append(label2id["I-OBJ"])  # Keep I-OBJ
            else:
                aligned_labels.append(label2id[current_label])
        prev_word_id = word_id

    encoding["labels"] = aligned_labels
    encoding["relation_label"] = example["relation_label"]
    return encoding

tokenized_data = [tokenize_and_align(ex) for ex in bio_data]

# Dataset class
class JointPersonaDataset(Dataset):
    def __init__(self, encodings):
        self.data = encodings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"]),
            "relation_label": torch.tensor(item["relation_label"])
        }

# Improved model with weighted loss
class JointBertExtractor(nn.Module):
    def __init__(self, base_model='bert-base-uncased', num_token_labels=4, num_relation_labels=0):
        super(JointBertExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, num_token_labels)
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size, num_relation_labels)

    def forward(self, input_ids, attention_mask, labels=None, relation_label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        cls_output = self.dropout(outputs.pooler_output)

        token_logits = self.token_classifier(sequence_output)
        relation_logits = self.relation_classifier(cls_output)

        loss = None
        if labels is not None and relation_label is not None:
            # Weighted loss for better object boundary detection
            class_weights = torch.tensor([1.0, 2.0, 3.0, 2.5]).to(labels.device)  # O, B-SUB, B-OBJ, I-OBJ
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            
            token_loss = loss_fct(token_logits.view(-1, token_logits.shape[-1]), labels.view(-1))
            relation_loss = nn.CrossEntropyLoss()(relation_logits, relation_label)
            loss = token_loss + relation_loss

        return {
            "loss": loss,
            "token_logits": token_logits,
            "relation_logits": relation_logits
        }

# Setup training
dataset = JointPersonaDataset(tokenized_data)
train_size = int(TRAIN_SPLIT * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JointBertExtractor(
    num_token_labels=len(label2id),
    num_relation_labels=len(relation2id)
).to(device)

# Improved optimizer settings
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

def compute_token_f1(preds, labels):
    true_labels = []
    true_preds = []
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_labels.append(l)
                true_preds.append(p)
    return f1_score(true_labels, true_preds, average="macro")

# Training loop
train_losses = []
val_f1s = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            relation_label=batch["relation_label"]
        )

        loss = outputs["loss"]
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            token_logits = outputs["token_logits"]
            preds = torch.argmax(token_logits, dim=-1)

            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(batch["labels"].cpu().tolist())

    f1 = compute_token_f1(val_preds, val_labels)
    val_f1s.append(f1)

    print(f"[Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | Val F1: {f1:.4f}")

# Save model
model_save_path = "src/llm/PExtractor"
os.makedirs(model_save_path, exist_ok=True)
tokenizer.save_pretrained(model_save_path)
torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
print(f"Model saved to {model_save_path}")

# Improved extraction function
def extract_triplet_joint(sentence: str, model, tokenizer, id2label, id2relation, device="cpu"):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    token_preds = torch.argmax(outputs["token_logits"], dim=-1).squeeze().cpu().tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    attention_mask = inputs["attention_mask"].squeeze().cpu().tolist()

    subject = None
    obj_tokens = []
    in_object = False

    for token, label_id, mask in zip(tokens_decoded, token_preds, attention_mask):
        if mask == 0 or token in ["[PAD]", "[CLS]", "[SEP]"]:
            continue
        
        label = id2label.get(label_id, "O")
        
        if label == "B-SUB":
            subject = token
        elif label == "B-OBJ":
            in_object = True
            obj_tokens.append(token)
        elif label == "I-OBJ" and in_object:
            obj_tokens.append(token)
        elif in_object and label == "O":
            break

    rel_pred_id = torch.argmax(outputs["relation_logits"], dim=-1).item()
    relation = id2relation[rel_pred_id]
    object_str = tokenizer.convert_tokens_to_string(obj_tokens).strip()
    subject = subject if subject else "i"

    return (subject, relation, object_str)
