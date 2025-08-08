import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, AdamW
from tqdm import tqdm

# === Dataset ===
class PIKDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        with open(path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }

    def __len__(self):
        return len(self.samples)

# === Wrapper with linear P(IK) head ===
class GemmaWithPIK(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        # Freeze all model parameters
        for param in self.base.parameters():
            param.requires_grad = False

        hidden_size = self.base.config.hidden_size  # usually 5120
        self.pik_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.base.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        cls_repr = last_hidden[:, -1, :]          # use last token: [B, H]
        pik_score = self.pik_head(cls_repr).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            loss = nn.BCELoss()(pik_score, labels)

        return {"loss": loss, "pik": pik_score}

# === Training loop ===
def train():
    model_id = "google/gemma-3-12b-it"
    data_path = "pik_sample_full.jsonl"
    batch_size = 2
    epochs = 5
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = PIKDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GemmaWithPIK(model_id).to(device)
    optimizer = AdamW(model.pik_head.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask, labels)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train()
