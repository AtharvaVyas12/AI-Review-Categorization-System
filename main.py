"""
=============================================================================
Project Title: AI-Based Review Categorization System
Description: A Multi-Class Sentiment Analysis Engine for User Reviews.
Author: [Your Name]
Enrollment No: [Your ID]
Version: 1.0.0
=============================================================================

This module provides a robust, enterprise-grade sentiment analysis pipeline.
It categorizes unstructured textual reviews into five distinct business classes:
[BEST, GOOD, NEUTRAL, BAD, POOR] using Natural Language Processing (NLP).

Features included:
    - Advanced Text Pre-processing (Noise reduction, regex filtering)
    - Lexicon-based Polarity Scoring
    - Batch Processing for CSV datasets
    - Built-in Flask REST API for web integration
    - Comprehensive Logging and Error Handling
"""

import os
import re
import csv
import sys
import json
import logging
import argparse
from datetime import datetime
from collections import Counter

# Third-party imports
try:
    from textblob import TextBlob
    from flask import Flask, request, jsonify
except ImportError:
    print("CRITICAL ERROR: Missing dependencies.")
    print("Please run: pip install textblob flask")
    sys.exit(1)


# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

class Config:
    """System-wide configuration and threshold constants."""
    
    # Sentiment Thresholds
    THRESHOLDS = {
        "BEST": 0.70,
        "GOOD": 0.15,
        "NEUTRAL_LOWER": -0.15,
        "NEUTRAL_UPPER": 0.15,
        "BAD": -0.60
    }
    
    # Categories
    CAT_BEST = "BEST"
    CAT_GOOD = "GOOD"
    CAT_NEUTRAL = "NEUTRAL"
    CAT_BAD = "BAD"
    CAT_POOR = "POOR"
    
    # Application Config
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    LOG_FILE = "system_logs.log"


# ==========================================
# LOGGING SETUP
# ==========================================

def setup_logger():
    """Configures the system logger for debugging and audit trails."""
    logger = logging.getLogger("AI_Review_System")
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File Handler
    fh = logging.FileHandler(Config.LOG_FILE)
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logger()


# ==========================================
# EXCEPTIONS
# ==========================================

class NLPProcessingError(Exception):
    """Custom exception raised for errors during text processing."""
    pass

class BatchProcessingError(Exception):
    """Custom exception raised for errors during CSV batch operations."""
    pass


# ==========================================
# CORE NLP MODULES
# ==========================================

class TextCleaner:
    """
    Handles the pre-processing and normalization of raw text data.
    Ensures that the sentiment engine receives clean, mathematical strings.
    """
    
    def __init__(self):
        log.debug("TextCleaner initialized.")

    def remove_urls(self, text):
        """Strips HTTP/HTTPS links from the text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_special_characters(self, text):
        """Removes punctuation and non-alphabetic characters."""
        return re.sub(r'[^A-Za-z\s]', '', text)

    def normalize_whitespace(self, text):
        """Removes extra spaces, tabs, and newlines."""
        return re.sub(r'\s+', ' ', text).strip()

    def process(self, raw_text):
        """
        Executes the full text cleaning pipeline.
        
        Args:
            raw_text (str): The original, unformatted review.
            
        Returns:
            str: The cleaned and lowercased string.
        """
        if not isinstance(raw_text, str):
            log.warning(f"Expected string for cleaning, got {type(raw_text)}. Casting to string.")
            raw_text = str(raw_text)
            
        try:
            text = self.remove_urls(raw_text)
            text = self.remove_special_characters(text)
            text = self.normalize_whitespace(text)
            return text.lower()
        except Exception as e:
            log.error(f"Error during text cleaning: {e}")
            raise NLPProcessingError("Failed to clean text data.")


class SentimentEngine:
    """
    The core algorithm class. Evaluates mathematical polarity and 
    maps it to the five distinct business categories.
    """
    
    def __init__(self):
        self.cleaner = TextCleaner()
        log.debug("SentimentEngine initialized.")

    def calculate_polarity(self, text):
        """Calculates the sentiment score from -1.0 to 1.0."""
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def map_category(self, score):
        """
        Maps a continuous floating-point score to a discrete business category.
        """
        if score >= Config.THRESHOLDS["BEST"]:
            return Config.CAT_BEST
        elif Config.THRESHOLDS["GOOD"] <= score < Config.THRESHOLDS["BEST"]:
            return Config.CAT_GOOD
        elif Config.THRESHOLDS["NEUTRAL_LOWER"] < score < Config.THRESHOLDS["NEUTRAL_UPPER"]:
            return Config.CAT_NEUTRAL
        elif Config.THRESHOLDS["BAD"] < score <= Config.THRESHOLDS["NEUTRAL_LOWER"]:
            return Config.CAT_BAD
        else:
            return Config.CAT_POOR

    def analyze(self, raw_review):
        """
        The main pipeline method. Cleans text, calculates score, and categorizes.
        
        Args:
            raw_review (str): The raw input string.
            
        Returns:
            dict: A payload containing the original text, clean text, score, and category.
        """
        log.info("Analyzing new review string.")
        try:
            cleaned_text = self.cleaner.process(raw_review)
            score = self.calculate_polarity(cleaned_text)
            category = self.map_category(score)
            
            result = {
                "original_text": raw_review,
                "cleaned_text": cleaned_text,
                "polarity_score": round(score, 4),
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
            log.debug(f"Analysis complete: {category} ({score})")
            return result
        except Exception as e:
            log.error(f"Sentiment Analysis failed: {e}")
            return {"error": str(e)}


# ==========================================
# BATCH PROCESSING MODULE
# ==========================================

class BatchProcessor:
    """
    Handles bulk analysis of datasets (e.g., CSV files).
    Useful for training evaluations and massive database dumps.
    """
    
    def __init__(self):
        self.engine = SentimentEngine()
        log.debug("BatchProcessor initialized.")

    def process_csv(self, input_filepath, output_filepath, text_column_index=0):
        """
        Reads a CSV, analyzes a specific column, and exports the results.
        """
        log.info(f"Starting batch process for file: {input_filepath}")
        
        if not os.path.exists(input_filepath):
            raise BatchProcessingError(f"File not found: {input_filepath}")

        results = []
        stats = Counter()

        try:
            with open(input_filepath, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader, None)
                
                for row_num, row in enumerate(reader, start=1):
                    if len(row) <= text_column_index:
                        log.warning(f"Row {row_num} skipped: insufficient columns.")
                        continue
                        
                    raw_text = row[text_column_index]
                    analysis = self.engine.analyze(raw_text)
                    
                    if "error" not in analysis:
                        stats[analysis["category"]] += 1
                        results.append(analysis)

            # Exporting to output CSV
            self._export_to_csv(output_filepath, results)
            self._print_summary(stats, len(results))
            
            return True

        except Exception as e:
            log.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(str(e))

    def _export_to_csv(self, filepath, data):
        """Helper method to write dictionary list to CSV."""
        if not data:
            log.warning("No data to export.")
            return

        keys = data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        log.info(f"Batch results saved successfully to {filepath}")

    def _print_summary(self, stats, total):
        """Prints a statistical summary of the batch job."""
        print("\n" + "="*40)
        print(" BATCH PROCESSING SUMMARY ")
        print("="*40)
        print(f"Total Reviews Processed : {total}")
        for category, count in stats.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f" - {category:<10}: {count:04d} ({percentage:.2f}%)")
        print("="*40 + "\n")


# ==========================================
# REST API MODULE (FLASK)
# ==========================================

app = Flask(__name__)
api_engine = SentimentEngine()

@app.route('/', methods=['GET'])
def health_check():
    """API Root - Checks if the server is running."""
    return jsonify({
        "status": "online",
        "service": "AI Review Categorization API",
        "version": "1.0.0"
    }), 200

@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    """
    API Endpoint for analyzing a single review.
    Expects JSON payload: {"review": "Text goes here"}
    """
    data = request.get_json()
    
    if not data or 'review' not in data:
        return jsonify({"error": "Missing 'review' field in JSON payload"}), 400
        
    review_text = data['review']
    result = api_engine.analyze(review_text)
    
    if "error" in result:
        return jsonify(result), 500
        
    return jsonify(result), 200

def run_server():
    """Boots up the Flask web server."""
    log.info(f"Starting Flask API Server on port {Config.API_PORT}...")
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=False)


# ==========================================
# COMMAND LINE INTERFACE (CLI)
# ==========================================

def run_cli_tests():
    """Runs a hardcoded set of tests for rapid demonstration."""
    print("\n--- Running System Diagnostics & Tests ---\n")
    engine = SentimentEngine()
    
    test_cases = [
        "This is the most amazing, life-changing software I have ever used. Absolutely perfect!",
        "The product arrived on time and works as expected. Good job.",
        "It's okay. Nothing special, but it gets the job done I suppose.",
        "The interface is confusing and it crashed twice today. Not happy.",
        "Absolutely atrocious service. The support team was rude and my account was deleted. I demand a refund!"
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case [{i}/5]:")
        result = engine.analyze(test)
        print(f" Input    : {test}")
        print(f" Score    : {result['polarity_score']}")
        print(f" Category : {result['category']}")
        print("-" * 50)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="AI Review Categorization System")
    
    parser.add_argument('--mode', choices=['test', 'server', 'batch'], default='test',
                        help="Operation mode: 'test' (run local tests), 'server' (start API), 'batch' (process CSV)")
    
    parser.add_argument('--input', type=str, help="Input CSV file path (Required for batch mode)")
    parser.add_argument('--output', type=str, default='output_results.csv', help="Output CSV file path (For batch mode)")
    
    args = parser.parse_args()

    # Routing based on CLI arguments
    if args.mode == 'test':
        run_cli_tests()
        
    elif args.mode == 'server':
        run_server()
        
    elif args.mode == 'batch':
        if not args.input:
            print("ERROR: --input file path is required for batch mode.")
            sys.exit(1)
        processor = BatchProcessor()
        try:
            processor.process_csv(args.input, args.output)
        except Exception as e:
            print(f"Batch execution failed: {e}")


if __name__ == "__main__":
    main()