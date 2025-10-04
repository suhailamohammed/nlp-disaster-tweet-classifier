# Project Overview
A Machine Learning pipeline to detect disaster-related tweets using text embeddings (BERT / Word2Vec) and Logistic Regression. 
This project is my submission to the [Kaggle NLP Competition](https://www.kaggle.com/competitions/nlp-getting-started/overview) for the disaster tweet classification problem.

## 🏆 Leaderboard Performance ([Leaderboard](https://www.kaggle.com/competitions/nlp-getting-started/leaderboard#))
- **Kaggle Rank:** 188 / 575 participants  
- **Percentile:** Top ~33%
  

## Project Structure
```
nlp-disaster-tweet-classifier/
│
├── dataset/                     # Folder containing all dataset CSV files
│   ├── train.csv                # Training dataset with labels
│   ├── test.csv                 # Test dataset for predictions
│   └── sample_submission.csv    # Sample submission format
│
├── dataset_loader.py            # Functions to load train/test datasets and sample submission
├── preprocessing.py             # Text cleaning functions (e.g., remove hashtags)
├── embeddings.py                # Convert tweets into BERT or Word2Vec embeddings
├── model.py                     # Train/test model functions
├── main.py                      # Main script to run full pipeline
└── requirements.txt             # Python dependencies
```

## Dataset
- **Training Data:** Labeled tweets with target `0` (not disaster) or `1` (disaster)  
- **Test Data:** Unlabeled tweets for prediction  
- **Sample Submission:** Format for submitting predictions  

[Kaggle dataset link](https://www.kaggle.com/competitions/nlp-getting-started/data)

## Features

- Clean and preprocess tweet text

- Convert tweets into embeddings:

- BERT embeddings for contextual representation

- Word2Vec embeddings for word-level semantic representation

- Train a Logistic Regression classifier

- Generate predictions for unseen tweets

- Export results in Kaggle-style submission format

## Instructions

Follow these steps to set up the environment and run the code:

---

1.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download and Prepare Dataset**:
    Download the dataset from **Kaggle**, extract the zip file, and store the contents in the **`dataset/`** folder.

4.  **Run the main.py script**:
    ```bash
    python main.py
    ```
    This will first load the dataset, then run and **train the model**, generate predictions, and finally save the results to the **`submission.csv`** file.



## Result Discussion
It was observed that using Word2Vec embeddings for training produced faster results, achieving an F1 score of approximately `0.79129`. However, when the BERT embedding method was applied, 
the F1 score increased significantly to `0.81274`. In my opinion, even though BERT required more time to process and generate embeddings, the performance improvement justified the additional computational cost.
