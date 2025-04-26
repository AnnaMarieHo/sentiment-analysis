import ollama
import re
import pandas as pd
import os
import concurrent.futures
import json

# Define target emotion list and the full emotion schema
EMOTION_COLUMNS = [
    'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'disappointment', 'disapproval', 'disgust',
    'excitement', 'fear', 'gratitude', 'joy', 'love', 'optimism', 'pride',
    'remorse', 'sadness', 'surprise','neutral'
]

# TARGET_EMOTIONS = ['remorse', 'fear', 'disgust', 'disappointment', 'approval', 'anger']
# TARGET_EMOTIONS = ['remorse']
TARGET_EMOTIONS = ['neutral']

# SEEDS = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 5086]  
# SEEDS = [124, 234, 345, 456, 567, 678, 789, 890, 901, 508]  
# SEEDS = [1244, 2344, 3454, 4564, 5674, 6784, 7894, 8904, 9014, 5084]  
# SEEDS = [12454, 23454, 34554, 45654, 56754, 67854, 78954, 89054, 90154, 50854]  
# SEEDS = [126554, 236554, 346554, 456554, 567554, 678554, 789554, 890554, 901554, 508554]  
# SEEDS = [128754, 238754, 348754, 458754, 568754, 678754, 788754, 898754, 908754, 508754]  
# SEEDS = [12854, 23854, 34854, 45854, 56854, 67854, 78854, 89854, 90854, 50854]  
# SEEDS = [12854, 23854, 34854, 45854, 56854, 67854, 78854, 89854, 90854, 50854]  
# SEEDS = [128541, 238541, 348541, 458541, 568541, 678541, 788541, 898541, 908541, 508541]  
# SEEDS = [108541, 208541, 308541, 408541, 508541, 608541, 708541, 808541, 908541, 108541]  
# SEEDS = [1085442, 2085442, 3085442, 4085442, 5085442, 6085442, 7085442, 8085442, 9085442, 1085442]  
# SEEDS = [1065442, 2065442, 3065442, 4065442, 5065442, 6065442, 7065442, 8065442, 9065442, 1065442]  
SEEDS = [1065432, 2065432, 3065432, 4065432, 5065432, 6065432, 7065432, 8065432, 9065432, 1065432]  

def generate_examples_for_emotion(emotion, num_samples, seed):
    """Use WizardLM to generate synthetic comments for a single emotion."""
    prompt = f"""Generate {num_samples} nuanced Reddit-style comments that express the emotion. Do not make them verbose or theatrical. This means that it is good to use slang and realistic language. **{emotion}**.

Only return results in this format:
Comment: "Your realistic sentence"
Label: {emotion}

Do NOT return anything else. Each comment should be entirely {emotion}.
Use natural, varied tone — avoid repetitive phrasing/repetition, exaggeration, or lists.
Do NOT use emotionally charged language.
Avoid including caring tone in these neutral comments.

below are example comments:

General Discussions
"The game launched in 2015 and has had three major updates."

"You can find that option in the settings menu under 'Privacy'."

"This subreddit has a rule against posting personal information."

Tech & Gaming Threads
"Your CPU usage seems normal for that workload."

"The PS5 supports both physical and digital copies of games."

"Linux distros vary in terms of ease of use and customization options."

Casual Conversations
"Yeah, that's how the tax system works in most countries."

"I think that's just how the algorithm ranks posts."

"It depends on what specs you're looking for in a laptop."

Help & Advice Threads
"Try clearing your cache and restarting the app."

"That model usually gets around 10 hours of battery life."

"You can file a support ticket through their website."

Science & Facts
"Water is denser than ice, which is why ice floats."

"The nearest star system to Earth is Alpha Centauri."

"Gravity affects all objects equally regardless of mass."
"""
    try:
        response = ollama.chat(
            model="gemma3:4b",
            options={"temperature": 0.4, "seed": seed},
            messages=[{"role": "user", "content": prompt}]
        )
        return emotion, seed, response["message"]["content"]
    except Exception as e:
        print(f"Error generating {emotion} with seed {seed}: {e}")
        return emotion, seed, ""

def parse_response(response_text):
    """Parse text to extract quotes and use them as neutral comments."""
    lines = response_text.splitlines()
    structured_data = []
    
    # Process each line and extract quoted text
    for line in lines:
        print("LINE: ", line)
        
        # Look for text within quotes
        # Match text inside double quotes
        pattern = r'["](.*?)["]' 
        matches = re.findall(pattern, line)
        
        for match in matches:
            comment = match.strip()
            # Basic length check
            if len(comment) > 5: 
                print("FOUND COMMENT: ", comment)
                # Create a row with the neutral label
                row = {e: 0 for e in EMOTION_COLUMNS}
                row["text"] = comment
                # Set neutral to 1
                row["neutral"] = 1 
                structured_data.append(row)
    
    print(f"Found {len(structured_data)} comments")
    return structured_data

def quality_check(comment, emotion):
    # Filter out comments that are too short
    if len(comment) < 10:
        return False
        
    # Filter out repetitive or low-quality comments
    low_quality_patterns = [
        # Simple "I am happy" patterns
        r'^\s*[Ii] (feel|am) (so|really) \w+\s*$',  
        # "That's so sad" patterns
        r'^\s*[Tt]hat\'?s so \w+\s*$',  
        # "This is so disgusting" patterns
        r'^\s*[Tt]his (is|makes me) (so|really) \w+\s*$',  
    ]
    
    for pattern in low_quality_patterns:
        if re.match(pattern, comment):
            return False
            
    return True

def save_to_csv(rows, filename="neutral_comments1.csv"):
    if not rows:
        print("No rows to save")
        return
        
    df = pd.DataFrame(rows)

    # Remove duplicates based on the 'text' column
    df = df.drop_duplicates(subset=["text"])

    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["text"])
        except pd.errors.EmptyDataError:
            print("Existing file was empty, create new file")
    
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} total rows ({len(rows)} new rows) to {filename}")

def process_batch(emotion_seed_pairs, batch_size=30):
    """Process a batch of emotion-seed pairs for generation."""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  
        future_to_pair = {
            executor.submit(generate_examples_for_emotion, emotion, batch_size, seed): (emotion, seed)
            for emotion, seed in emotion_seed_pairs
        }
        
        for future in concurrent.futures.as_completed(future_to_pair):
            emotion, seed, response_text = future.result()
            if response_text:
                print(f"Generated {emotion} with seed {seed}")
                results.append((emotion, seed, response_text))
            else:
                print(f"Failed to generate {emotion} with seed {seed}")
    
    return results

def initialize_batches():
    """Create all emotion-seed pairs for processing."""
    pairs = []
    for emotion in TARGET_EMOTIONS:
        for seed in SEEDS:
            pairs.append((emotion, seed))
    return pairs

def track_progress(filename="generation_progress.json"):
    """Track which emotion-seed pairs have been processed."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {"processed": [], "counts": {emotion: 0 for emotion in TARGET_EMOTIONS}}

def update_progress(progress, emotion, seed, count, filename="generation_progress.json"):
    """Update the progress tracker."""
    # Convert tuple to list for JSON serialization if needed
    emotion_seed = [emotion, seed]
    if emotion_seed not in progress["processed"]:
        progress["processed"].append(emotion_seed)
        progress["counts"][emotion] += count
    
    with open(filename, 'w') as f:
        json.dump(progress, f)
    
    return progress

if __name__ == "__main__":
    # Force reprocessing by initializing fresh progress
    progress = {"processed": [], "counts": {emotion: 0 for emotion in TARGET_EMOTIONS}}
    
    # Process all pairs
    all_pairs = initialize_batches()
    # Process all pairs regardless of previous runs
    remaining_pairs = all_pairs  
    
    if not remaining_pairs:
        print("All emotion-seed pairs have been processed")
    else:
        print(f"Processing {len(remaining_pairs)} emotion-seed pairs")
        
    all_rows = []
    
    # Process remaining pairs in batches
    if remaining_pairs:
        batch_results = process_batch(remaining_pairs)
        print(f"Got {len(batch_results)} batch results")
        
        # Parse the results
        for emotion, seed, response_text in batch_results:
            print(f"\nParsing results for: {emotion} (seed: {seed})")
            print("Sample response: " + response_text[:100] + "...")
            parsed = parse_response(response_text)
            
            # Update progress
            update_progress(progress, emotion, seed, len(parsed))
            
            all_rows.extend(parsed)
    
    # Save all collected rows to a single CSV file
    print(f"Total rows to save: {len(all_rows)}")
    save_to_csv(all_rows, "neutral_comments1.csv")
    
    # Print summary of dataset
    if os.path.exists("neutral_comments1.csv"):
        final_df = pd.read_csv("neutral_comments1.csv")
        print("\nDataset Summary:")
        for emotion in TARGET_EMOTIONS:
            count = final_df[final_df[emotion] == 1].shape[0]
            print(f"  - {emotion}: {count} samples")
        print(f"  Total unique comments: {final_df.shape[0]}")
