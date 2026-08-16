# 📧 Spam Email Detection

A machine learning project that classifies emails as **spam** or **ham (legitimate)** using NLP text processing and classification algorithms, deployed as an interactive Streamlit web app.

## 🔍 Overview

This project builds an end-to-end spam email classifier — from raw email text to a working prediction app. It uses TF-IDF vectorization to convert email text into numerical features, then compares three different classification algorithms to select the best-performing model.

## 📊 Dataset

- **Source:** [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) (Kaggle)
- **Size:** ~5,171 emails (after removing duplicates)
- **Classes:** Spam (~29%) vs Ham (~71%)
- **Format:** Raw email text with `Subject:` lines, based on the Enron email corpus

## 🛠️ Approach

1. **Data Cleaning:** Lowercasing, removing `Subject:` prefixes, numbers, punctuation, and extra whitespace
2. **Feature Extraction:** TF-IDF vectorization (top 3,000 features)
3. **Train-Test Split:** 80/20 split with stratification to preserve class balance
4. **Model Comparison:** Trained and evaluated 3 classifiers to compare precision/recall trade-offs

## 🤖 Model Comparison

| Model | Accuracy | Spam Precision | Spam Recall | False Positives |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **97.9%** | **0.96** | 0.97 | **13** |
| Random Forest | 97.5% | 0.94 | **0.98** | 20 |
| Naive Bayes | 94.2% | 0.87 | 0.94 | 43 |

**Logistic Regression was selected** as the final model — it achieved the highest accuracy and the fewest false positives (legitimate emails incorrectly flagged as spam), which is the more critical error to minimize in a real-world email filter, since losing a genuine email is generally worse than an occasional spam message slipping through.

## ⚠️ Known Limitation

The training data (Enron corpus) consists primarily of formal business emails. As a result, the model performs very reliably on similar formal/business-style text but can misclassify short, casual, conversational messages (e.g., simple greetings), since this style was underrepresented during training. This is a classic **domain mismatch** issue and highlights the importance of aligning training data distribution with real-world deployment use cases.

## 🚀 Running the App Locally

```bash
# Clone the repository
git clone https://github.com/riyauc19-creator/spam-email-detection.git
cd spam-email-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
spam-email-detection/
├── app.py                          # Streamlit web app
├── spam_classifier_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
├── requirements.txt                # Python dependencies
└── README.md
```

## 🧰 Tech Stack

- Python
- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Random Forest)
- pandas
- Streamlit
- joblib

## 📈 Future Improvements

- Expand training data to include more casual/conversational email examples to reduce domain mismatch
- Hyperparameter tuning (GridSearchCV) for further performance gains
- Add explainability (e.g., highlight which words drove the spam prediction)
