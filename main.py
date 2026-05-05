# AI-Based Review Categorization System
# Multi-Class Sentiment Analysis

from textblob import TextBlob
import re

def clean_text(text):
    """Removes special characters and standardizes text."""
    text = re.sub(r'[^A-Za-z\s]', '', str(text))
    return text.lower()

def multi_class_categorize(review_text):
    """Categorizes review into 5 buckets based on sentiment polarity."""
    cleaned_text = clean_text(review_text)
    
    # Calculate Polarity via NLP Lexicon (-1.0 to 1.0)
    score = TextBlob(cleaned_text).sentiment.polarity
    
    if score >= 0.70:
        category = "BEST"
    elif 0.15 <= score < 0.70:
        category = "GOOD"
    elif -0.15 < score < 0.15:
        category = "NEUTRAL"
    elif -0.60 < score <= -0.15:
        category = "BAD"
    else:
        category = "POOR"
        
    return {"text": review_text, "score": round(score, 3), "category": category}

# --- Test Execution ---
if __name__ == "__main__":
    print("--- AI Review Analyzer Initialized ---")
    samples = [
        "This software is phenomenal! It completely transformed our workflow. 10/10 highly recommended.",
        "The product works well and arrived on time.",
        "It's okay, but the user interface is clunky and confusing.",
        "Disgusting service. The item was broken, customer care hung up on me. Total scam."
    ]
    
    for review in samples:
        result = multi_class_categorize(review)
        print(f"\nReview: '{review}'\nAssigned Category: [{result['category']}] (Score: {result['score']})")