# 📰 Fake News Detection System

A machine learning pipeline to classify news articles as **real** or **fake** using NLP preprocessing, multiple vector embeddings, and a range of ML/DL models with hyperparameter tuning.

---

## 📁 Project Structure

```
fake_news_detection_system/
│
├── data/
│   ├── fake_news_dataset.csv        # Raw dataset
│   ├── data.csv                     # Cleaned dataset (text + label)
│   ├── metadata.csv                 # Article metadata (title, source, author, etc.)
│   └── training_data/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
│
├── models/
│   ├── best_ml_model.pkl            # Best sklearn model (joblib)
│   ├── best_dl_model.keras          # Best deep learning model (Keras)
│   ├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│   └── w2v_model.bin                # Saved Word2Vec model
│
├── notebooks/
│   └── preprocessing.ipynb          # Full pipeline notebook
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw Text
   │
   ├── Remove Punctuation
   ├── Lowercase
   ├── Tokenization
   ├── Stopword Removal  (NLTK)
   ├── Stemming          (PorterStemmer)
   └── Lemmatization     (WordNetLemmatizer)
         │
         ├── TF-IDF Vectorizer  (max 5000 features)
         └── Word2Vec Embeddings (vector_size=100)
               │
               └── ML / DL Model Training + Evaluation
```

---

## 🧠 Models Tested

### Machine Learning (with GridSearchCV + 5-Fold StratifiedKFold)
| Model | Tuned Parameters |
|---|---|
| Logistic Regression | C, solver |
| Random Forest | n_estimators, max_depth, min_samples_split |
| SVM | C |
| Decision Tree | max_depth, min_samples_split |
| KNN | n_neighbors, weights |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| XGBoost | n_estimators, learning_rate, max_depth |
| MLP (sklearn) | hidden_layer_sizes, alpha |

### Deep Learning (with EarlyStopping)
| Architecture | Layers |
|---|---|
| DL-128-64 | Dense(128) → Dropout → Dense(64) → Sigmoid |
| DL-256-128 | Dense(256) → Dropout → Dense(128) → Sigmoid |
| DL-512-256-128 | Dense(512) → Dropout → Dense(256) → Dense(128) → Sigmoid |

---

## 📊 Embeddings

- **TF-IDF** — Bag-of-words frequency weighting (sparse, 5000 features)
- **Word2Vec** — Dense semantic embeddings (100-dimensional, averaged per document)

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Harsh0patel/fake_news_detection_system.git
cd fake_news_detection_system
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 🚀 Usage

Open and run `notebooks/preprocessing.ipynb` from top to bottom. The notebook will:

1. Load and clean the dataset
2. Apply full NLP preprocessing
3. Build TF-IDF and Word2Vec embeddings
4. Train and tune all ML models using GridSearchCV + KFold
5. Train all DL architectures with early stopping
6. Print a final results table sorted by accuracy
7. Automatically save the best model to `models/`

---

## 💾 Loading the Best Model

```python
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load('models/best_ml_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Predict on new text
sample = ["government announces new economic policy"]
X = tfidf.transform(sample)
prediction = model.predict(X)
print("real" if prediction[0] == 1 else "fake")
```

For the best deep learning model:
```python
from tensorflow.keras.models import load_model

model = load_model('models/best_dl_model.keras')
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
nltk
gensim
xgboost
tensorflow
joblib
matplotlib
seaborn
```

---

## 👤 Author

**Harsh Patel**
[github.com/Harsh0patel](https://github.com/Harsh0patel)