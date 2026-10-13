# 📊 YouTube Comment Sentiment Analysis

An end-to-end **NLP and Machine Learning web application** that fetches public YouTube comments for any video and classifies them as **Positive, Neutral or Negative**, with an interactive analytics dashboard.

The model is a **TF-IDF + Logistic Regression** pipeline trained on **~1 million labelled YouTube comments**, deployed with **Streamlit**.

---

## 📸 Application Preview

### 🎬 Video Input & Details

![Home](https://github.com/cpravin2110/Youtube-comment-sentiment-analysis/blob/6f5596d57198be6204b9841b25da71e5be7dfece/Img/Screenshot%202026-10-13%20110839.png)

### 📊 Sentiment Dashboard

![Dashboard](https://github.com/cpravin2110/Youtube-comment-sentiment-analysis/blob/6f5596d57198be6204b9841b25da71e5be7dfece/Img/Screenshot%202026-10-13%20110858.png)

### 🥧 Sentiment Distribution & Counts

![Distribution](https://github.com/cpravin2110/Youtube-comment-sentiment-analysis/blob/6f5596d57198be6204b9841b25da71e5be7dfece/Img/Screenshot%202026-10-13%20110928.png)

### ☁️ Word Cloud

![Word Cloud](https://github.com/cpravin2110/Youtube-comment-sentiment-analysis/blob/6f5596d57198be6204b9841b25da71e5be7dfece/Img/Screenshot%202026-10-13%20111052.png)

### 💬 Comment Highlights & CSV Export

![Comments](https://github.com/cpravin2110/Youtube-comment-sentiment-analysis/blob/6f5596d57198be6204b9841b25da71e5be7dfece/Img/Screenshot%202026-10-13%20111113.png)

---

## 🚀 Live Demo

👉 **[Open Live Demo](https://youtube-comment-sentiment-analysis-pravinchavan.streamlit.app/)**

---

## 🏷️ Technologies

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-85C1E9)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)

---

## 📌 Project Overview

**YouTube Comment Sentiment Analysis** lets a user paste any YouTube video URL, choose how many comments to analyse, and instantly see:

1. 🎥 **Video details**: title, channel, publish date, views, likes, total comments
2. 📊 **Sentiment breakdown**: Positive / Neutral / Negative counts and percentages
3. ☁️ **Word cloud** of the most frequent words
4. 💬 **Top 5 positive and negative comments**
5. 📥 **Downloadable CSV** of all analysed comments

ML pipeline:

**Data Cleaning → Train/Test Split → TF-IDF Vectorization → Model Comparison → Hyperparameter Tuning → Final Model → Serialization → Streamlit Deployment**

---

## 📌 Highlights

- Trained on **1,032,225 labelled YouTube comments** from **4,563 videos** (Kaggle dataset).
- Perfectly balanced classes (~33% each), so **accuracy and macro F1** are reliable metrics.
- Built a reusable **scikit-learn Pipeline** (TF-IDF + Logistic Regression).
- Used **unigram + bigram TF-IDF** with 100,000 features and sublinear TF scaling.
- Compared **7 classifiers** using **stratified 3-fold cross-validation**.
- Ran **GridSearchCV** (12 combinations × 3 folds = 36 fits) on TF-IDF and regularization settings.
- Final model reached **70.84% test accuracy** and **0.71 macro F1** on 206,413 unseen comments.
- Beat the Multinomial Naive Bayes baseline (**66.3%**) by about 4.5 points.
- Saved the full pipeline with **Joblib** and loaded it in the Streamlit app.
- Added a dashboard with charts, word cloud, comment highlights and CSV export.

---

# 🖥️ App Features

| Section | Description |
|---|---|
| **🎬 Video Input** | Paste a YouTube URL and set the maximum number of comments to analyse (slider). |
| **🎥 Video Details** | Title, channel, publish date, views, likes and total comments. |
| **📊 Sentiment Analysis** | Total analysed, plus Positive / Negative / Neutral counts with percentages. |
| **🥧 Visualisations** | Sentiment distribution pie chart, sentiment counts bar chart and sentiment-over-time view. |
| **☁️ Word Cloud** | Most frequent words across the analysed comments. |
| **💬 Comment Highlights** | Top 5 positive and top 5 negative comments. |
| **📥 Export** | Download all analysed comments (text, sentiment, timestamp, likes) as CSV. |

---

# 📍 Dataset

**Source:** [YouTube Comments Sentiment Dataset (Kaggle)](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset)

| Property | Value |
|---|---|
| Total comments | 1,032,225 |
| Rows after cleaning | 1,032,064 |
| Unique videos | 4,563 |
| Time span | April 2013 – February 2025 |
| Columns | 12 |
| Target | `Sentiment` (Positive / Neutral / Negative) |

### Class Distribution

| Sentiment | Comments | Share |
|---|---:|---:|
| Negative | 346,075 | 33.53% |
| Positive | 343,317 | 33.26% |
| Neutral | 342,672 | 33.21% |

### Columns

`CommentID`, `VideoID`, `VideoTitle`, `AuthorName`, `AuthorChannelID`, `CommentText`, `Sentiment`, `Likes`, `Replies`, `PublishedAt`, `CountryCode`, `CategoryID`

Only `CommentText` (input) and `Sentiment` (target) were used for modelling.

---

# 🧠 Machine Learning Workflow

```text
        ┌──────────────────────────┐
        │  Kaggle Dataset (1.03M)  │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Data Cleaning & Checks   │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Stratified 80/20 Split   │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ TF-IDF (1-2 grams)       │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ 7 Classifiers + 3-Fold CV│
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ GridSearchCV Tuning      │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Final Logistic Regression│
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Joblib → Streamlit App   │
        └──────────────────────────┘
```

---

# ⚙️ Machine Learning Process

## 1. Data Cleaning

- Checked missing values (only 631 missing author names, which were not used)
- Checked full-row duplicates (0 found)
- Removed **161** empty / whitespace-only comments
- Standardised sentiment labels and kept only the 3 valid classes
- Parsed timestamps (0 invalid)
- Analysed comment-length distribution (median 65 characters, 95th percentile 326)

## 2. Train / Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| Split | Size |
|---|---:|
| Training | 825,651 |
| Testing | 206,413 |

Stratification keeps the class balance identical in both sets.

## 3. Text Preprocessing

A lightweight function decodes HTML entities, replaces URLs with a `URL` token, and normalises whitespace. Case, punctuation and emojis are preserved because they carry sentiment signal in YouTube comments.

## 4. Feature Extraction: TF-IDF

```python
TfidfVectorizer(
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
```

Unigrams + bigrams capture phrases such as "not good" that single words miss.

---

# 🤖 Models Compared

Stratified **3-fold CV** on a 100,000-comment sample from the training set, scored by **macro F1**:

| Rank | Model | CV Accuracy | CV Macro F1 | Time (min) |
|---:|---|---:|---:|---:|
| 🥇 1 | **Logistic Regression** | **0.6641** | **0.6651** | 0.70 |
| 2 | Ridge Classifier | 0.6589 | 0.6596 | 0.46 |
| 3 | SGD Classifier | 0.6557 | 0.6553 | 0.35 |
| 4 | Linear SVM | 0.6511 | 0.6517 | 0.50 |
| 5 | Complement Naive Bayes | 0.6398 | 0.6347 | 0.31 |
| 6 | Multinomial Naive Bayes | 0.6380 | 0.6334 | 0.36 |
| 7 | Random Forest | 0.6207 | 0.6216 | 39.75 |

**Takeaway:** Linear models outperform tree ensembles on sparse TF-IDF features, and Logistic Regression is also fast. Random Forest was the slowest (~40 min) and among the weakest.

---

# 🔧 Hyperparameter Tuning

`GridSearchCV` with `scoring="f1_macro"` and 3-fold stratified CV.

| Parameter | Values |
|---|---|
| `tfidf__max_features` | 50,000 / 100,000 |
| `tfidf__ngram_range` | (1,1) / (1,2) |
| `classifier__C` | 0.5 / 1.0 / 2.0 |

**Total:** 12 combinations × 3 folds = **36 fits**

**Best parameters:**

```python
{'classifier__C': 1.0, 'tfidf__max_features': 100000, 'tfidf__ngram_range': (1, 2)}
```

**Best CV Macro F1:** `0.6651`

The tuned configuration matched the original settings, which confirms the baseline setup was already well chosen.

---

# 📊 Final Results

The final pipeline was retrained on the **full training set** (825,651 comments) and evaluated on the untouched test set.

| Metric | Result |
|---|---:|
| **Test Accuracy** | **70.84%** |
| Macro Precision | 0.71 |
| Macro Recall | 0.71 |
| **Macro F1** | **0.71** |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.71 | 0.73 | 0.72 | 69,215 |
| Neutral | 0.65 | 0.68 | 0.66 | 68,534 |
| Positive | 0.78 | 0.72 | 0.75 | 68,664 |

### Performance Progression

| Stage | Model | Accuracy |
|---|---|---:|
| Baseline | Multinomial Naive Bayes | 66.30% |
| Tuned on 100k sample | Logistic Regression | 67.36% |
| **Final (full data)** | **Logistic Regression** | **70.84%** |

> **Note:** The 100k-sample model scored 67.36%. Training on all 825k comments added about 3.5 points, so **more data helped more than tuning**.

**Observation:** Positive is the easiest class (F1 0.75). Neutral is the hardest (F1 0.66) because neutral comments overlap with mild positive and negative language.

---

# 🔄 Application Pipeline

```text
   YouTube URL
        │
        ▼
 Extract Video ID
        │
        ▼
 Fetch Video Details + Comments
        │
        ▼
 Text Preprocessing
        │
        ▼
 TF-IDF → Logistic Regression
        │
        ▼
 Sentiment Labels
        │
        ▼
 Charts • Word Cloud • Highlights • CSV
```

---

# 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.12 |
| Data Analysis | Pandas, NumPy |
| NLP / ML | Scikit-learn, NLTK, TF-IDF |
| Models Tested | Logistic Regression, Ridge, SGD, Linear SVM, Naive Bayes, Random Forest |
| Visualisation | Matplotlib, WordCloud, Streamlit charts |
| Data Source | Kaggle (training), YouTube Data API (live comments) |
| Serialization | Joblib |
| Web Application | Streamlit |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |
| Development | Jupyter Notebook |

---

# 📁 Project Structure

```text
YouTube-Comment-Sentiment-Analysis/
│
├── app.py
├── models/
│   └── sentiment_pipeline.pkl
├── notebook/
│   └── Youtube-Comment-Analysis.ipynb
├── Img/
├── requirements.txt
└── README.md
```

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `models/sentiment_pipeline.pkl` | Trained TF-IDF + Logistic Regression pipeline |
| `notebook/` | Training, comparison and evaluation notebook |
| `Img/` | README screenshots |
| `requirements.txt` | Python dependencies |

---

# ▶️ Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/cpravin2110/YOUR-REPO.git
cd YOUR-REPO
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Add Your YouTube API Key

Create `.streamlit/secrets.toml`:

```toml
YOUTUBE_API_KEY = "your_api_key_here"
```

## 5. Run the App

```bash
streamlit run app.py
```

---

# 📦 Model & Dependency Compatibility

The pipeline was saved with **scikit-learn 1.9.1** (Python 3.12). Pin the same version in `requirements.txt` to avoid errors when loading the `.pkl` file.

---

# ☁️ Deployment

```text
Local Development → GitHub Repository → Streamlit Community Cloud → Live Web App
```

The API key is stored in Streamlit secrets and never committed to the repository.

---

# 🚧 Limitations & Future Improvements

### Current Limitations

- The model is trained mainly on **English** comments. **Hinglish / code-mixed** comments (common on Indian creators' videos) are often labelled Neutral, as seen in the gaming vlog demo.
- Bag-of-words models can't understand sarcasm, context or tone.
- Short comments and emoji-only comments carry little signal.
- Neutral is the hardest class (F1 0.66).
- Labels in the training dataset may be noisy.

### Future Improvements

- Fine-tune a **transformer** (BERT / RoBERTa / XLM-R) for higher accuracy
- Add **multilingual and Hinglish** support
- Add emoji-aware features
- Add reply-thread analysis and topic modelling
- Add **SHAP / LIME** explainability
- Compare sentiment across multiple videos or channels
- Cache API results to save quota

---

# 📌 Key Takeaways

- Working with a **~1M row** text dataset
- Data cleaning and validation
- Stratified train/test splitting
- Text preprocessing and **TF-IDF** with n-grams
- Comparing 7 classifiers using **stratified K-fold CV**
- **GridSearchCV** on a full ML pipeline
- Evaluation with accuracy, macro F1 and per-class metrics
- Balancing **accuracy vs training time**
- Model serialization with **Joblib**
- Building and deploying a **Streamlit** dashboard
- Working with an external API

---

# 📊 Final Model Summary

| Metric | Result |
|---|---:|
| Training Data | 1,032,064 comments |
| Videos | 4,563 |
| Classes | 3 (balanced) |
| Models Compared | 7 |
| Cross-Validation | 3-Fold Stratified |
| Grid Search Fits | 36 |
| Final Model | **TF-IDF + Logistic Regression** |
| Test Accuracy | **70.84%** |
| Macro F1 | **0.71** |
| Deployment | **Streamlit Community Cloud** |

---

# 👨‍💻 Author

## Pravin Chavan

**Data Science | Machine Learning | GenAI Enthusiast**

Final-year B.E. Computer Engineering Student

### Connect With Me

🔗 **LinkedIn:** https://www.linkedin.com/in/iampravinchavan

💻 **GitHub:** https://github.com/cpravin2110

🌐 **Portfolio:** https://cpravin2110.github.io/PravinPortfolio/

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ **Star** on GitHub.

Thank you for checking out the project! 🚀
