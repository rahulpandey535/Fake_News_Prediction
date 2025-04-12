# Fake_News_Prediction

# 📰 Fake News Detection using Logistic Regression

This project focuses on building a **Fake News Detection** system using **Natural Language Processing (NLP)** and **Logistic Regression**. It classifies news articles as **real** or **fake** based on their content using TF-IDF vectorization.

---

## 📌 Features

- Combines `author` and `title` to form content.
- Applies custom **text preprocessing** including stemming.
- Uses **TF-IDF** with bigram support and stopword removal.
- Trained using **Logistic Regression** with 5-fold cross-validation.
- Saves both the model and vectorizer using `joblib` and `pickle`.
- Includes a script for **making predictions on new articles**.

---

## 🧠 Tech Stack

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `nltk` for preprocessing
  - `scikit-learn` for modeling and evaluation
  - `joblib` and `pickle` for model serialization

---

## 📊 Model Performance

| Metric                   | Score     |
|--------------------------|-----------|
| Training Accuracy        | 99.2%     |
| Test Accuracy            | 53.7%     |
| Mean Cross-Validation    | ~56-60%   |

> Note: The test accuracy indicates that while the model is likely **overfitting**, further optimization can help generalize better.

---

## 🗂️ Project Structure

. ├── news.csv # Dataset with title, author, label ├── model_training.py / .ipynb # Main model training script ├── predict_news.py # Script to classify new article ├── logistic_regression_model.pkl # Trained Logistic Regression model (joblib) ├── logistic_regression_model.sav # Trained Logistic Regression model (pickle) ├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer (joblib) ├── tfidf_vectorizer.sav # Saved TF-IDF vectorizer (pickle) └── README.md # Documentation

---

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
2. Install Requirements
pip install -r requirements.txt
3. Train the Model (Optional)
python model_training.py
4. Predict New News
python predict_news.py
📈 Improvements to Try

Use more advanced models (e.g., XGBoost, BERT).
Implement GridSearch for hyperparameter tuning.
Explore external datasets for improved generalization.
Use more feature-rich inputs (e.g., article body, source).
👨‍💻 Author

Rahul Pandey
🎓 Computer Science Student | Aspiring Data Scientist & iOS Developer
📬 Let's connect: LinkedIn
📝 License

This project is under the MIT License.
