
import re
import json
import torch
import pandas as pd
from torch.nn import Sigmoid
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel


EMOTION_COLUMNS = [
    'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust',
    'excitement', 'fear', 'gratitude', 'joy', 'love', 'optimism', 'pride',
    'remorse', 'sadness', 'surprise', 'neutral'
]

def get_embeddings_batch(texts, model_path="./fine_tuned_roberta_multi_label_w_neutral_v2", batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    sigmoid = Sigmoid()
    all_embeddings = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=250,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = sigmoid(logits).cpu().numpy()

            # Extract [CLS] embedding from last hidden layer
            hidden_states = outputs.hidden_states[-1]  # last layer
            cls_embeddings = hidden_states[:, 0, :]  # [CLS] token

        for j in range(len(batch_texts)):
            embedding = cls_embeddings[j].cpu().numpy().tolist()
            all_embeddings.append(embedding)

            # Map emotion probs to emotion names
            emotion_probs = dict(zip(EMOTION_COLUMNS, probs[j]))
            all_probs.append(emotion_probs)

    return all_embeddings, all_probs



def remove_links(text):
    # Regular expression to identify URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def extract_post_and_comments_cleaned(json_file_path):
    # Read the JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract post data as a dictionary
    post_data = data.get('post', {})
    
    # Extract comments as a list of dictionaries
    comments_data = data.get('comments', [])
    
    # Create DataFrames
    post_df = pd.DataFrame([post_data]) if post_data else pd.DataFrame()
    comments_df = pd.DataFrame(comments_data) if comments_data else pd.DataFrame()
    
    # Clean the text fields by removing links
    if not post_df.empty:
        post_df = post_df.applymap(lambda x: remove_links(x) if isinstance(x, str) else x)
    if not comments_df.empty:
        comments_df = comments_df.applymap(lambda x: remove_links(x) if isinstance(x, str) else x)

    # Extract all comment bodies
    # print(post_df)
    all_comments = comments_df['body'].tolist()
    if post_df['body'].tolist() == '':
        post = post_df['title'].tolist()
    else:
        post = post_df['body'].tolist()
    # Get embeddings and emotion probabilities in batches
    embeddings_list, emotion_probs_list = get_embeddings_batch(all_comments, batch_size=10)
    post_embeddings_list, post_emotion_probs_list = get_embeddings_batch(post, batch_size=10)
    # Store in DataFrame
    post_df['embeddings'] = post_embeddings_list  # Initialize first
    post_df['emotion_probs'] = post_emotion_probs_list  # Safe to assign as a column directly
    
    comments_df['embeddings'] = None  # Initialize first
    for idx, embedding in enumerate(embeddings_list):
        comments_df.at[idx, 'embeddings'] = embedding

    comments_df['emotion_probs'] = emotion_probs_list  # Safe to assign as a column directly

    print(f"Successfully processed {len(all_comments)} comments in batches")
    
    return post_df, comments_df

# Usage
json_file_path = "./reddit_json/reddit_data12.json"
post_df, comments_df = extract_post_and_comments_cleaned(json_file_path)


post_df['emotion_probs'] = post_df['emotion_probs'].apply(lambda probs: {k: float(v) for k, v in probs.items()})
post_df.to_csv('post_df_with_embeddings.csv', index=False)
comments_df['emotion_probs'] = comments_df['emotion_probs'].apply(lambda probs: {k: float(v) for k, v in probs.items()})

comments_df.to_csv('comments_df_with_embeddings.csv', index=False)
