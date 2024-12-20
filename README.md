# Fake News Detection Application

This project is a **Fake News Detection Application** that uses machine learning to classify news articles as "Fake" or "Real." It preprocesses the input data, extracts relevant features, and predicts the authenticity of news content through a trained logistic regression model. The application is implemented using Streamlit for an interactive user interface.

---

## Table of Contents
- [Features](#features)
- [Project Files](#project-files)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Interactive Interface**: Users can input a news title and content for real-time classification.
- **Text Preprocessing**: Removes noise, punctuation, and stopwords, and applies lemmatization for better feature extraction.
- **Machine Learning Model**: Utilizes logistic regression with TF-IDF and Count Vectorizer techniques.
- **Modular Design**: Separates preprocessing, feature extraction, and prediction functions for maintainability.
- **Streamlit Deployment**: Offers an intuitive web-based interface for user interaction.

---

## Project Files

1. **Python Script**: `fake_news_detectioin_app.py`
   - Main application script to run the Streamlit app.
   - Contains preprocessing, prediction logic, and user interface components.

2. **Notebook**: `Fake_news_prediction.ipynb`
   - Jupyter Notebook that documents the model training and testing pipeline.
   - Includes data exploration, preprocessing, vectorization, and model evaluation.

3. **Trained Models**:
   - Logistic Regression Model: `trained_model_for_fake_news_detection.sav`
   - TF-IDF Vectorizer: `tfidfmodel.sav`
   - Count Vectorizer: `countvectmodel.sav`

---

## Technologies Used

- **Languages**: Python
- **Libraries**:
  - Data Preprocessing: `nltk`, `re`, `string`
  - Machine Learning: `pickle`
  - Deployment: `streamlit`
- **Pre-trained Models**:
  - Logistic Regression for classification
  - TF-IDF and Count Vectorizer for feature extraction

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```
3.  **Download NLTK Data:**

```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```
4. **Place Model Files:**
    Copy ```trained_model_for_fake_news_detection.sav```, ```tfidfmodel.sav```, and ```countvectmodel.sav``` to the appropriate directory.
5. **Run the Streamlit App:**
```bash
streamlit run fake_news_detectioin_app.py
```
# Usage
1.  Open the Streamlit app in your browser.
2.  Input the title and content of the news article.
3.  Click the "Predict fake or Real News" button.
4.  View the prediction result: "Fake News" or "True News."

# How It Works
1. Input:
   - Users enter the title and content of a news article.
2. Preprocessing:
   - The input text is cleaned, tokenized, and lemmatized.
   - Stopwords and unnecessary characters are removed.
3. Feature Extraction:
   - Uses TF-IDF and Count Vectorizer models for text representation.
4. Prediction:
   - The logistic regression model classifies the news as fake or real.
5. Output:
   - The result is displayed on the app interface.

# Future Enhancements
- Multi-language Support: Expand preprocessing and classification for non-English news.
- Confidence Scores: Show the model's confidence in its predictions.
- Live News Analysis: Integrate APIs for real-time news scraping and analysis.
- Deep Learning: Explore transformer-based models like BERT for improved accuracy.

# Contributing
### Contributions are welcome! If you wish to contribute:

- Fork the repository.
- Create a new branch: ```git checkout -b feature-branch```.
- Commit your changes: ```git commit -m "Add a new feature"```.
- Push to the branch: ```git push origin feature-branch```.
- Submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
