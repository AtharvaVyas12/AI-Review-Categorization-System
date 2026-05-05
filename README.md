# AI-Based Review Categorization System (Multi-Class Sentiment Analysis)

## 📌 Project Overview
This project is a robust Natural Language Processing (NLP) pipeline designed to automatically categorize unstructured user reviews into five distinct business categories: `BEST`, `GOOD`, `NEUTRAL`, `BAD`, and `POOR`. 

Unlike standard binary classifiers, this engine utilizes a weighted lexicon approach via `TextBlob` and rigorous text-preprocessing to achieve multi-class granularity. It features a modular architecture, batch CSV processing capabilities, and a built-in Flask REST API.

## ⚙️ Features
* **Modular OOP Architecture:** Clean separation of concerns (TextCleaners, SentimentEngines).
* **Multi-Class Categorization:** 5 distinct threshold-based output buckets.
* **REST API:** Built-in Flask server for easy frontend integration.
* **Batch Processing:** Ability to ingest large CSV files, analyze them, and export analytical reports.

## 🚀 How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run Diagnostics / Local Test**
```bash
python main.py --mode test
```

**3. Run the API Server**
```bash
python main.py --mode server
```

**4. Process a Dataset**
```bash
python main.py --mode batch --input dataset.csv --output results.csv
```
