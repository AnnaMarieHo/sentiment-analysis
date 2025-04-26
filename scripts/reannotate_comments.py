import pandas as pd
import ollama
import os
import re
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
goemotions = pd.read_csv("goemotions_1.csv")

INSTRUCTION_TEMPLATE = """
Perform multiclass sentiment analysis on the following comments.
Identify exactly 8 present emotions from the given categories, considering tone and context.

**YOU MUST SELECT EMOTIONS ONLY FROM THIS EXACT LIST. DO NOT CREATE OR GUESS NEW EMOTIONS.**  
**Allowed Emotions (STRICTLY ONE-WORD ONLY):**
"Agreement", "Aggression", "Amusement", "Anger", "Annoyance", "Anticipation", "Anxiety", "Apology",
"Approval", "Caring", "Confidence", "Contempt", "Confusion", "Curiosity", "Defiance", "Disappointment",
"Disbelief", "Disgust", "Disapproval","Disagreement", "Encouragement", "Excitement", "Fear", "Frustration", "Gratitude", "Hope",
"Love", "Joy", "Neutral", "Pride", "Resignation", "Resilience", "Sarcasm", "Sadness", "Skepticism", "Support",
and "Surprise".

---

### **Instructions:**
- **Select exactly 8 emotions. NO MORE, NO LESS.**
- **STRICTLY PICK FROM THE LIST ABOVE. DO NOT INVENT NEW EMOTIONS.**
- **Do NOT assign emotions randomly.** Analyze tone and sarcasm carefully.
- **Do NOT reuse emotions**
- **Aggressive language does NOT mean hostility.** Recognize when it is supportive.
- **Always detect supportive and encouraging emotions** when applicable.
- **If a match is unclear, choose the closest equivalent from the list.**
- **Do not choose neutral unless absolutely necessary.**
- **Do NOT ASSIGN neutral if a comment already has 5 categories.**
- **DO NOT ADD EXPLANATIONS OR EXTRA WORDS. ONLY RETURN THE EMOTION ARRAY.**
- **DO NOT ADD EMOTIONS THAT ARE NOT IN THE LIST. CHOOSE THE CLOSEST EQUIVALENT FROM THE LIST.**

---

Example 1 (Aggressive but Supportive)
Comment: "You better push yourself harder, or else you’re just wasting your time!" 

[Aggression, Defiance, Encouragement, Confidence, Frustration, Support, Skepticism, Resignation, Annoyance, Pride]

Example 2 (Hostile Aggression Without Support)
Comment: "You’re an idiot if you believe that nonsense. Wake up!"

[Aggression, Anger, Disgust, Contempt, Frustration, Defiance, Skepticism, Disappointment, Sarcasm, Annoyance]

Example 3 (Encouraging Despite Harsh Language)
Comment: "Get over it and prove them wrong. Stop complaining and make things happen!" 

[Aggression, Encouragement, Confidence, Support, Defiance, Resilience, Annoyance, Amusement, Skepticism, Pride]

Example 4 (Your Case)
Comment: "You do right, if you don't care then fuck 'em!" 

[Aggression, Defiance, Encouragement, Support, Frustration, Skepticism, Resignation, Amusement, Confidence, Pride]

Example 5 (Cautious Optimism with Encouragement)
Comment: "I really hope this works out for you. Just don’t get your hopes too high." 

[Hope, Encouragement, Skepticism, Anticipation, Anxiety, Support, Confidence, Neutral, Resilience, Caring]

Example 6 (Sarcastic Support - Fake Encouragement)
Comment: "Shockingly, some people continue to educate and improve themselves. You might benefit from it." 

[Sarcasm, Skepticism, Contempt, Disappointment, Disbelief, Annoyance, Confidence, Frustration, Amusement, Resignation]

Example 7 (Nostalgic Reflection on the Past)
Comment: "I still remember those late-night talks and how we thought we had everything figured out. Feels like a lifetime ago."

[Sadness, Love, Resignation, Pride, Hope, Curiosity, Support, Anxiety]

Example 8 (Deep Contemplation on Life’s Uncertainty)
Comment: "Funny how life never turns out the way you expect. Makes you wonder if we ever really have control over anything."

[Confusion, Skepticism, Disbelief, Resignation, Sadness, Curiosity, Frustration, Anticipation]

Example 9 (Hesitant Compassion - Caring but Cautious)
Comment: "I want to believe you, but I’ve been let down before. Still, I hope things turn out okay."

[Caring, Skepticism, Hope, Anxiety, Disappointment, Support, Sadness, Anticipation]

Example 10 (Overwhelming Awe - Stunned but Grateful)
Comment: "I can’t believe I’m standing here, in this moment. It’s more than I ever dreamed of."

[Surprise, Gratitude, Excitement, Hope, Love, Pride, Anticipation, Confidence]

---
**RETURN AN ARRAY OF EMOTIONS!!!**
**ALWAYS PLACE EMOTIONS IN BRACKETS!!!**
**DO NOT RETURN EMOTIONS WITHOUT BRACKETS!!!**
### **FORMAT RULES (STRICTLY FOLLOW THIS TEMPLATE)**:
Your output **MUST** follow this structure exactly:

Comment: "Example Comment" 
-
-
[Emotion1, Emotion2, Emotion3, Emotion4, Emotion5, Emotion6, Emotion7, Emotion8]


"""



def get_user_feedback(current_instruction):
    """This was designed to allow user to provide additional feedback without removing the original guidelines.
       It became more of a debugging strategy to interupt the annotation process when the model was producing 
       inaccurate annotations.
    """

    print("\nWould you like to add any refinements for the next batch?")
    print("1. Yes")
    print("2. No")
    
    # choice = input("Enter choice (1 or 2): ").strip()
    choice = "2"
    
    if choice == "1":
        print("\n Enter additional instructions (they will be **appended**, not replaced):")
        additional_guidelines = input("Additional instructions: ").strip()
        
        if additional_guidelines:
            updated_instruction = f"{current_instruction}\n\n### Additional Guidance:\n{additional_guidelines}"
            print("\n New guidelines have been **appended**.")
            return updated_instruction  
        else:
            print("\n No new input provided. Using previous instructions.")
            return current_instruction  

    return current_instruction  

def process_batch(start_idx, batch_size):
    """Process comments in batches with feedback adjustments"""
    
    global INSTRUCTION_TEMPLATE  # Declare global inside the function
    
    batch_results = []
    
    for i in range(start_idx, len(goemotions), batch_size):
        batch = goemotions.iloc[i:i+batch_size]["text"]
        
        batch_comments = [format_comment(comment) for comment in batch]
        
        response_text = label_comments(batch_comments, INSTRUCTION_TEMPLATE)
        print(response_text)
        
        emotion_lists = [match for match in response_text.splitlines() if re.search(r'\[.*\]$', match)]


        for comment, response in zip(batch_comments, emotion_lists):
            result = process_labels(response, comment)
            batch_results.append(result)
        
        save_to_csv(batch_results, filename="reannotated_goemotions.csv")
        batch_results.clear()  # Clear for next batch

        # Ask for user feedback before the next batch
        INSTRUCTION_TEMPLATE = get_user_feedback(INSTRUCTION_TEMPLATE)  # Pass and update outside

def format_comment(comment):
    """Ensure comments are properly formatted."""
    if pd.isna(comment):  
        return "No comment provided."
    return str(comment).strip().replace("\n", " ")  # Trim and remove newlines

def label_comments(comments, instruction):
    """ Sends batch of comments to the model for sentiment analysis """
    
    formatted_comments = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comments)])

    response = ollama.chat(
        model="wizardlm2:7b",
        # model="gemma3:4b", 
        options={  
            "seed": 50,  
            "temperature": 0.05 
        },
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\n**Comments to analyze:**\n{formatted_comments}"
        }]
    )

    return response['message']['content'].strip()

def load_emotion_embeddings():
    """Precomputes embeddings for the allowed emotion list to speed up similarity matching."""
    emotions = [
        "Agreement", "Aggression", "Amusement", "Anger", "Annoyance", "Anticipation", "Anxiety", "Apology",
        "Approval", "Caring", "Confidence", "Contempt", "Confusion", "Curiosity", "Defiance", "Disappointment",
        "Disbelief", "Disgust", "Disapproval","Disagreement","Encouragement", "Excitement", "Fear", "Frustration", "Gratitude", "Hope",
        "Love", "Joy", "Neutral", "Pride", "Resignation", "Resilience", "Sarcasm", "Sadness", "Skepticism", "Support",
        "Surprise"
    ]
    return {e: ollama.embeddings("nomic-embed-text", e)["embedding"] for e in emotions}
    # testing various sentence to vec models. Best was nomic-embed-text
    # return {e: ollama.embeddings("granite-embedding:278m", e)["embedding"] for e in emotions}
    # return {e: ollama.embeddings("mxbai-embed-large:335m", e)["embedding"] for e in emotions}

EMOTION_EMBEDDINGS = load_emotion_embeddings()



def closest_emotion(word):
    """Finds the most similar valid emotion using embeddings."""
    try:
        word_embedding = ollama.embeddings("nomic-embed-text", word)["embedding"]
        # word_embedding = ollama.embeddings("granite-embedding:278m", word)["embedding"]
        # word_embedding = ollama.embeddings("mxbai-embed-large:335m", word)["embedding"]
        similarities = {
            emotion: cosine_similarity([word_embedding], [embedding])[0][0]
            for emotion, embedding in EMOTION_EMBEDDINGS.items()
        }
        best_match = max(similarities, key=similarities.get)
        print(similarities[best_match])
        return best_match if similarities[best_match] >= 0.70 else None 
    except:
        return None  


import re
import difflib

def process_labels(response_text, formatted_comment):
    """Extracts emotion labels from model response while ensuring reasoning is separate."""
    # Define allowed emotions
    emotion_categories = set(EMOTION_EMBEDDINGS.keys())
    # Create mapping for case-insensitive matching
    standard_emotions_lower = {e.lower(): e for e in emotion_categories}

    detected_emotions = set()


    # print(response_text)
    bracket_matches = re.findall(r'\[.*?\]', response_text)
    # print(bracket_matches)
    # print("Valid emotions: ", standard_emotions_lower)

    reasoning = response_text.strip()  

    if bracket_matches and bracket_matches != '[NAME]':
        for match in bracket_matches:
            # Remove anything inside parentheses
            cleaned_emotions = re.sub(r'\([^)]*\)', '', match)
            # Remove square brackets around emotions  
            cleaned_emotions = cleaned_emotions.strip("[]")  
            emotions = [e.strip().lower() for e in cleaned_emotions.split(',') if e.strip()]

            # print(emotions)
            for emotion in emotions:
                # print(f"Checking emotion: '{emotion}'")
                if emotion in standard_emotions_lower:
                    # print(f"Direct match found for: {emotion}")
                    # simple textual similarity
                    detected_emotions.add(standard_emotions_lower[emotion])  
                    found_closest = True

                else:
                    
                    print(f"Emotion '{emotion}' is not valid, finding closest match...")
                    closest_by_embedding = closest_emotion(emotion)
                    print("emotion: ", emotion)
                    print("closest emotion: ", closest_by_embedding)
                    if closest_by_embedding:
                        detected_emotions.add(closest_by_embedding)
                        # print("Updated/substitute: ", detected_emotions)
                    found_closest = False

                        
                if found_closest is False:
                    closest_match = difflib.get_close_matches(emotion, standard_emotions_lower.keys(), n=1, cutoff=0.8)
                    print(closest_match)
                    if closest_match:
                        detected_emotions.add(standard_emotions_lower[closest_match[0]])
                        print("Corrected Spelling: ",detected_emotions)
    # reasoning = re.sub(r'\[.*?\]', '', reasoning).strip()

    label_dict = {emotion: (1 if emotion in detected_emotions else 0) for emotion in emotion_categories}
    label_dict["text"] = formatted_comment
    # label_dict["reasoning"] = reasoning  
    
    return label_dict



def save_to_csv(results, filename="reannotated_goemotions.csv", mode='a'):
    """Ensures new categorizations are appended to the same CSV file without overwriting previous data."""
    
    df = pd.DataFrame(results)  
    
    file_exists = os.path.isfile(filename)
    
    # if file_exists:
    #     existing_df = pd.read_csv(filename)  # Load existing data
    #     # Only append rows that are not already in the existing file
    #     df = df[~df["text"].isin(existing_df["text"])]  # Check for duplicates based on the "text" column
    #     # Append the non-duplicate rows
    #     df = pd.concat([existing_df, df], ignore_index=True)
    #     df.to_csv(filename, mode='w', index=False)  # Write back without overwriting headers
    # else:
    #     df.to_csv(filename, mode='w', index=False)  # Create a new file with headers if it doesn’t exist
    
    # print(f" Results appended to {filename} in a structured format!")

    if file_exists:
        # Load existing data
        existing_df = pd.read_csv(filename)  
        # df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=['text'], keep='last')

        df = pd.concat([existing_df, df], ignore_index=True)  # Append new data
        df.to_csv(filename, mode='w', index=False)  # Write back without overwriting headers
    else:
        df.to_csv(filename, mode='w', index=False)  # Create a new file with headers if it doesn’t exist
    
    print(f" Results appended to {filename} in a structured format!")


if __name__ == "__main__":
    process_batch(start_idx=44803, batch_size=2)  # Adjust batch_size as needed


