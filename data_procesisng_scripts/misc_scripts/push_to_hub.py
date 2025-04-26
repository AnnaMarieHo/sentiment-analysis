import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, create_repo

# Define the emotion columns 
EMOTION_COLUMNS = [
    'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust',
    'excitement', 'fear', 'gratitude', 'joy', 'love', 'optimism', 'pride',
    'remorse', 'sadness', 'surprise', 'neutral'
]

def push_to_hub(model_path, repo_name=None, token=None):

    # Push model to huggingface hub
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # If repo_name is not provided, create a default one
    if repo_name is None:
        model_name = os.path.basename(model_path)
        repo_name = f"{model_name}-emotion-classifier"
    
    print(f"Pushing model to HuggingFace Hub as '{repo_name}'...")
    
    # Create model card with information about emotions and usage
    model_card = f"""---"""
    
    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Set up metadata for the model config
    model.config.id2label = {i: label for i, label in enumerate(EMOTION_COLUMNS)}
    model.config.label2id = {label: i for i, label in enumerate(EMOTION_COLUMNS)}
    
    # Push to Hub
    try:
        # Create a new repo if it doesn't exist
        if token:
            api = HfApi(token=token)
            try:
                create_repo(repo_name, token=token, exist_ok=True)
            except Exception as e:
                print(f"Note: {e}")
                
        # Push the model and tokenizer
        model.push_to_hub(repo_name, use_auth_token=token)
        tokenizer.push_to_hub(repo_name, use_auth_token=token)
        
        print(f"Model successfully pushed to https://huggingface.co/{repo_name}")
        return f"https://huggingface.co/{repo_name}"
    except Exception as e:
        print(f"Error pushing model to hub: {e}")
        return None

if __name__ == "__main__":
    # Get model path 
    model_path = input("Enter the path to model (default: ../../fine_tuned_roberta_multi_label_w_neutral_v2): ")
    if not model_path.strip():
        model_path = "../../fine_tuned_roberta_multi_label_w_neutral_v2"
    
    # Get repository name
    repo_name = input("Enter repository name: ")
    if not repo_name.strip():
        repo_name = None
    
    # Get token
    token = input("Enter HuggingFace token: ")
    if not token.strip():
        token = None
    
    # Push to hub
    hub_url = push_to_hub(model_path, repo_name, token)
    
    if hub_url:
        print(f"\nModel pushed successfully: {hub_url}")
    else:
        print("\nFailed to push model to Hugging Face Hub.")