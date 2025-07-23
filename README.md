# Spam Classifier for Email and SMS

![Spam Classifier](https://img.shields.io/badge/ML-Spam%20Classifier-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Status](https://img.shields.io/badge/Status-Live-success)

## Live Demo
**Try the app:** [https://vivekspamclassifier.streamlit.app/](https://vivekspamclassifier.streamlit.app/)

## Overview
A machine learning-powered web application that classifies text messages as spam or legitimate (ham). Built using **Extra Trees Classifier** and **TF-IDF vectorization**, deployed on Streamlit Cloud for real-time spam detection.

## Features
- **Intelligent Classification**: Uses Extra Trees Classifier for accurate spam detection
- **Text Preprocessing**: Advanced NLP preprocessing with tokenization and stemming
- **Web Interface**: User-friendly Streamlit web application
- **Real-time Prediction**: Instant results for any text input
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack
- **Machine Learning**: scikit-learn (Extra Trees Classifier)
- **NLP**: NLTK (tokenization, stopwords removal, stemming)
- **Feature Extraction**: TF-IDF Vectorization
- **Web Framework**: Streamlit
- **Language**: Python 3.8+
- **Deployment**: Streamlit Cloud

## Model Performance
The Extra Trees Classifier was chosen for its:
- High accuracy in text classification
- Resistance to overfitting
- Fast training and prediction times
- Excellent performance on imbalanced datasets

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Vivektheprogrammer/Spam_Classifier.git
   cd Spam_Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## Usage
1. **Enter your message**: Type or paste any email/SMS content in the text area
2. **Click Predict**: Press the "Predict" button to analyze the message
3. **View Results**: Get instant classification as "Spam" or "Not Spam"

### Example Messages to Test:
- **Spam**: "Congratulations! You've won $1000! Click here to claim your prize now!"
- **Ham**: "Hey, are we still meeting for coffee tomorrow at 3 PM?"

## Project Structure
```
Spam_Classifier/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── spam.csv                        # Training dataset
├── etc_model.joblib                # Trained Extra Trees model
├── tfidf_vectorizer.joblib         # TF-IDF vectorizer
├── tfidf_vectorizer_revised.joblib # Revised vectorizer
├── Spam Classifier.ipynb           # Jupyter notebook with model development
├── Content for checking.txt        # Sample test messages
├── debug.log                       # Application logs
├── Images/                         # Screenshots and visualizations
│   ├── p1_intro.png               # Project introduction
│   ├── p2_Datacleaning.png        # Data cleaning process
│   ├── p4_EDA.png - p15_EDA.png   # Exploratory data analysis
│   └── P16_DP.png - P25_DP.png    # Data preprocessing steps
└── README.md                       # Project documentation
```

## How It Works

### 1. Text Preprocessing
- **Tokenization**: Breaks text into individual words
- **Normalization**: Converts to lowercase
- **Cleaning**: Removes punctuation and non-alphanumeric characters
- **Stopword Removal**: Eliminates common English stopwords
- **Stemming**: Reduces words to their root form using Porter Stemmer

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts preprocessed text to numerical features
- **Sparse Matrix**: Efficient storage of high-dimensional feature vectors

### 3. Classification
- **Extra Trees Classifier**: Ensemble method for robust classification
- **Binary Output**: 1 for Spam, 0 for Ham (Not Spam)

### Main Interface
The clean and intuitive interface allows users to easily input text and get predictions.

### Data Analysis Process
The project includes comprehensive exploratory data analysis and preprocessing steps documented in the Images folder.

## Model Development
The complete model development process is documented in `Spam Classifier.ipynb`, including:
- Data exploration and visualization
- Text preprocessing pipeline
- Model training and evaluation
- Performance metrics analysis
- Feature importance analysis

## Configuration
The application uses pre-trained models stored as joblib files:
- `etc_model.joblib`: Extra Trees Classifier model
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer for feature extraction

## Deployment
The application is deployed on **Streamlit Cloud** and accessible at:
**[https://vivekspamclassifier.streamlit.app/](https://vivekspamclassifier.streamlit.app/)**

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author
**Vivek R** - *The Programmer*
- GitHub: [@Vivektheprogrammer](https://github.com/Vivektheprogrammer)
- Project Link: [https://github.com/Vivektheprogrammer/Spam_Classifier](https://github.com/Vivektheprogrammer/Spam_Classifier)

---
  **If you found this project helpful, please give it a star on GitHub!** 
