
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the emotion columns (same as in your training script)
EMOTION_COLUMNS = [
    'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust',
    'excitement', 'fear', 'gratitude', 'joy', 'love', 'optimism', 'pride',
    'remorse', 'sadness', 'surprise', 'neutral'
]

def predict_emotions(text, model_path="../fine_tuned_roberta_multi_label_w_neutral_v2", threshold=0.30):
    """
    Predict emotions for a single text comment.
    
    Args:
        text (str): The text to analyze
        model_path (str): Path to the trained model
        threshold (float or list): Threshold(s) for prediction. Can be a single value or list of thresholds per emotion.
        
    Returns:
        dict: Predicted emotions and their probabilities
    """
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Use custom thresholds if available
    try:
        # Did not finish implementing and testing this feature
        thresholds_df = pd.read_csv("test_results/optimal_thresholds.csv")
        thresholds = thresholds_df['threshold'].tolist()
        print("Using optimal thresholds for each emotion")
    except:
        # Fall back to single threshold if file not found
        if isinstance(threshold, float):
            thresholds = [threshold] * len(EMOTION_COLUMNS)
        else:
            thresholds = threshold
        print(f"Using default threshold: {threshold}")
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
    
    # Convert to numpy for easier handling
    probs = probabilities.cpu().numpy()[0]
    
    # Get predicted emotions using thresholds
    predictions = {}
    predicted_emotions = []
    
    for i, emotion in enumerate(EMOTION_COLUMNS):
        prob = float(probs[i])
        predictions[emotion] = prob
        if prob >= thresholds[i]:
            predicted_emotions.append(emotion)
    
    return {
        "text": text,
        "probabilities": predictions,
        "predicted_emotions": predicted_emotions,
        "thresholds_used": dict(zip(EMOTION_COLUMNS, thresholds))
    }

def print_predictions(result):
    """Print prediction results in a readable format"""
    print("\n===== Emotion Analysis =====")
    print(f"Text: \"{result['text']}\"")
    print("\nDetected emotions:")
    
    if not result['predicted_emotions']:
        print("  No emotions detected above threshold")
    else:
        # Sort emotions by probability
        emotions = [(e, result['probabilities'][e]) for e in result['predicted_emotions']]
        emotions.sort(key=lambda x: x[1], reverse=True)
        
        for emotion, prob in emotions:
            threshold = result['thresholds_used'][emotion]
            print(f"  - {emotion.upper()} ({prob:.4f}, threshold: {threshold:.2f})")
    
    print("\nAll emotion probabilities:")
    # Sort all emotions by probability for display
    all_emotions = [(e, p) for e, p in result['probabilities'].items()]
    all_emotions.sort(key=lambda x: x[1], reverse=True)
    
    for emotion, prob in all_emotions:
        threshold = result['thresholds_used'][emotion]
        indicator = "✓" if prob >= threshold else " "
        print(f"  {indicator} {emotion}: {prob:.4f} (threshold: {threshold:.2f})")
    
    print("===========================\n")

if __name__ == "__main__":
    # Get input from user
    print("Enter a comment to analyze emotions (or 'quit' to exit):")
    
    while True:
        user_input = input("> ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_input.strip():
            print("Please enter a comment.")
            continue
            
        # Analyze the comment
        result = predict_emotions(user_input)
        print_predictions(result)