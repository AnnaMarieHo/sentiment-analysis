import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# config 
EMOTION_COLUMNS = [
    'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust',
    'excitement', 'fear', 'gratitude', 'joy', 'love', 'optimism', 'pride',
    'remorse', 'sadness', 'surprise', 'neutral'
]
MODEL_NAME = "roberta-base"
DATA_PATH = "./final_train_dataset.csv"
BATCH_SIZE = 16
EPOCHS = 10
MAX_LENGTH = 256

#  load data
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(len(df))
# Identify underrepresented classes
class_counts = df[EMOTION_COLUMNS].sum()
underrepresented = class_counts[class_counts < class_counts.median()].index.tolist()
print(f"Underrepresented classes: {underrepresented}")


# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_COLUMNS),
    problem_type="multi_label_classification"
)

# dataset class
class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe[EMOTION_COLUMNS].values.astype("float32")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# split data
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, 
                                   stratify=df[EMOTION_COLUMNS].values)
train_dataset = EmotionDataset(train_df)
val_dataset = EmotionDataset(val_df)

# metrics
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    # Add per-class F1 scores
    class_f1 = {}
    for i, cls in enumerate(EMOTION_COLUMNS):
        class_f1[f"f1_{cls}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)

    metrics = {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }

    # Add class-specific metrics
    metrics.update(class_f1)
    
    # Save metrics to a text file
    with open("training_metrics.txt", "a") as f:
        import datetime
        f.write(f"\n{'='*50}\n")
        f.write(f"Evaluation at {datetime.datetime.now()}\n")
        f.write(f"{'='*50}\n")
        
        # Write overall metrics
        f.write("Overall Metrics:\n")
        for key in ["micro_f1", "macro_f1", "accuracy", "precision", "recall"]:
            f.write(f"{key}: {metrics[key]:.4f}\n")
        
        # Write per-class F1 scores
        f.write("\nPer-class F1 Scores:\n")
        for cls in EMOTION_COLUMNS:
            f.write(f"f1_{cls}: {metrics[f'f1_{cls}']:.4f}\n")
        
        f.write("\n")
    
    return metrics

# training args
training_args = TrainingArguments(
    output_dir="./emotion_roberta_w_neutral_v2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,  # Lower learning rate
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=10,  # More epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",  # Focus on balanced performance
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,  # Use mixed precision if your GPU supports it
    warmup_ratio=0.1,  # Add warmup
)

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # If alpha is None, it will be set in the forward method based on class frequencies
        self.alpha = alpha

    def forward(self, logits, targets):
        # Calculate class weights if alpha wasn't provided
        if self.alpha is None:
            # Compute class frequencies from targets
            pos_weight = targets.sum(dim=0)
            neg_weight = targets.shape[0] - pos_weight
            # Compute the weight for positive examples (inverse frequency weighting)
            pos_weight = torch.where(pos_weight > 0, 
                                    targets.shape[0] / (2 * pos_weight), 
                                    torch.ones_like(pos_weight))
        else:
            pos_weight = self.alpha
            
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss term
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply class weights to positive examples
        weight = targets * pos_weight.unsqueeze(0) + (1 - targets)
        
        # Final loss
        loss = weight * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BalancedFocalLossTrainer(Trainer):
    def __init__(self, *args, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = WeightedFocalLoss(alpha=None, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# trainer
trainer = BalancedFocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    focal_gamma=4.0,  # Increase gamma to focus more on hard examples
)

# train
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    trainer.train()

    # save model
    model.save_pretrained("./fine_tuned_roberta_multi_label_w_neutral_v2")
    tokenizer.save_pretrained("./fine_tuned_roberta_multi_label_w_neutral_v2")
    print("Model and tokenizer saved to ./fine_tuned_roberta_multi_label_w_neutral_v2")

