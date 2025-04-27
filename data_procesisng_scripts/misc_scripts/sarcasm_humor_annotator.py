import pandas as pd
import ollama
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
# This is for later identification of sarcastic and humorous comments. Not used in this project.
# Instruction template focused on sarcasm and humor
SARCASM_HUMOR_TEMPLATE = """
Analyze the following comments to identify and categorize sarcasm and humor types.
Focus specifically on detecting the presence and type of sarcasm or humor in each comment.

**Categories to consider:**
1. **Sarcasm Types:**
   - "Deadpan": Delivered with a serious, expressionless tone
   - "Self-deprecating": Sarcasm directed at oneself
   - "Exaggerated": Obvious overstatement for effect
   - "Bitter": Expressing disappointment or frustration
   - "Mocking": Ridiculing or making fun of something/someone

2. **Humor Types:**
   - "Wordplay": Puns, double entendres, etc.
   - "Absurdist": Illogical or nonsensical humor
   - "Observational": Commenting on everyday situations
   - "Dark": Dealing with unpleasant subjects
   - "Reference": Alluding to pop culture, memes, etc.
   - "Situational": Humor arising from specific contexts

**Output Format:**
For each comment, classify whether it contains sarcasm, humor, both, or neither.
If present, specify the type(s) from the categories above.

---

### Instructions:
- Analyze tone, context, and linguistic markers carefully
- Consider cultural references and internet slang/expressions
- If sarcasm is present, identify which type it is
- If humor is present, identify which type it is
- Comments can have multiple types or none at all
- If neither sarcasm nor humor is present, label as "Neither"

### Example Outputs:

Comment: "Oh great, another meeting that could have been an email."
[Sarcasm: Bitter, Humor: Observational]

Comment: "I'm so talented I can turn coffee into bad decisions."
[Sarcasm: Self-deprecating, Humor: Observational]

Comment: "The wifi is being so fast today I've only refreshed the page 47 times."
[Sarcasm: Exaggerated, Humor: Observational]

Comment: "Thank you for your detailed explanation of a topic I've been studying for 20 years."
[Sarcasm: Mocking]

Comment: "I stayed up all night wondering where the sun went, then it dawned on me."
[Humor: Wordplay]

Comment: "I'm just here to share information about the upcoming community event."
[Neither]

---

**FORMAT RULES:**
Your output for each comment must follow this structure:

Comment: "{text}"
[Category: Type, Category: Type] OR [Neither]
"""

def get_user_feedback(current_instruction):
    """This was designed to allow user to provide additional feedback without removing the original guidelines.
       It became more of a debugging strategy to interupt the annotation process when the model was producing 
       inaccurate annotations.
    """
    print("\nWould you like to add any refinements for the next batch?")
    print("1. Yes")
    print("2. No")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nEnter your additional instructions (they will be **appended**, not replaced):")
        additional_guidelines = input("Additional instructions: ").strip()
        
        if additional_guidelines:
            updated_instruction = f"{current_instruction}\n\n### Additional Guidance:\n{additional_guidelines}"
            print("\nInstructions updated! New guidelines have been **appended**.")
            return updated_instruction  
        else:
            print("\nNo new input provided. Using previous instructions.")
            return current_instruction  

    return current_instruction  

def process_batch(dataset_path, start_idx=0, batch_size=5, output_file="sarcasm_humor_annotations.csv"):
    """Process comments in batches with feedback adjustments"""
    
    global SARCASM_HUMOR_TEMPLATE
    
    # Load dataset
    try:
        dataset = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with {len(dataset)} comments.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    batch_results = []
    
    for i in range(start_idx, len(dataset), batch_size):
        # Adjust column name if needed
        batch = dataset.iloc[i:i+batch_size]["text"]  
        
        print(f"Processing batch {i//batch_size + 1}, comments {i+1}-{min(i+batch_size, len(dataset))}")
        
        batch_comments = [format_comment(comment) for comment in batch]
        
        response_text = label_comments(batch_comments, SARCASM_HUMOR_TEMPLATE)
        print("\nModel Response:")
        print(response_text)
        
        # Extract classifications from the response
        classifications = extract_classifications(response_text, batch_comments)
        
        for comment, classification in zip(batch, classifications):
            result = {
                "text": comment,
                "classification": classification
            }
            batch_results.append(result)
        
        # Save after each batch
        save_to_csv(batch_results, filename=output_file)
        
        # Ask for user feedback before the next batch
        if i + batch_size < len(dataset):
            SARCASM_HUMOR_TEMPLATE = get_user_feedback(SARCASM_HUMOR_TEMPLATE)

def format_comment(comment):
    """Ensure comments are properly formatted."""
    if pd.isna(comment):  
        return "No comment provided."
    # Trim and remove newlines
    return str(comment).strip().replace("\n", " ")  

def label_comments(comments, instruction):
    """Sends batch of comments to the model for sarcasm/humor analysis"""
    
    formatted_comments = "\n\n".join([f"Comment: \"{comment}\"" for comment in comments])

    response = ollama.chat(
        # adjust the model as needed
        model="wizardlm2:7b", 
        # Lower temperature for more consistent outputs 
        options={  
            "seed": 50,  
            "temperature": 0.1  
        },
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\n**Comments to analyze:**\n{formatted_comments}"
        }]
    )

    return response['message']['content'].strip()

def extract_classifications(response_text, batch_comments):
    """Extract classifications from model response"""
    
    classifications = []
    
    # Split response by comment
    # Skip the first empty element
    comment_blocks = re.split(r'Comment: ".*?"', response_text)[1:]  
    
    # If unable to split properly, try another approach
    if len(comment_blocks) != len(batch_comments):
        # Look for classification brackets directly
        bracket_patterns = re.findall(r'\[(.*?)\]', response_text)
        return bracket_patterns if len(bracket_patterns) == len(batch_comments) else ["Error: Couldn't parse response"] * len(batch_comments)
    
    for block in comment_blocks:
        match = re.search(r'\[(.*?)\]', block)
        if match:
            classifications.append(match.group(0))
        else:
            classifications.append("Error: No classification found")
    
    return classifications

def save_to_csv(results, filename="sarcasm_humor_annotations.csv"):
    """Save results to CSV file"""
    
    df = pd.DataFrame(results)
    
    file_exists = os.path.isfile(filename)
    
    if file_exists:
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    # usage
    print("Sarcasm and Humor Annotation")
    print("--------------------------------")
    
    dataset_path = input("Enter the path to dataset (CSV file): ")
    start_idx = int(input("Enter starting index (0 for beginning): "))
    batch_size = int(input("Enter batch size: "))
    output_file = input("Enter output filename (default: sarcasm_humor_annotations.csv): ") or "sarcasm_humor_annotations.csv"
    
    process_batch(dataset_path, start_idx, batch_size, output_file) 