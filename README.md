# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated machine learning system for detecting fraudulent credit card transactions in real-time using advanced algorithms and geospatial analysis.

## 🎯 Project Overview

This project implements a comprehensive fraud detection system that analyzes credit card transactions using multiple features including:
- **Geospatial Analysis**: Distance calculation between customer and merchant locations
- **Temporal Features**: Transaction timing patterns (hour, day, month)
- **Categorical Data**: Merchant categories and customer demographics
- **Amount Analysis**: Transaction value patterns

The system uses **LightGBM** (Light Gradient Boosting Machine) for robust fraud detection with high accuracy and low false positive rates.

## ✨ Features

- 🔍 **Real-time Fraud Detection**: Instant analysis of transaction data
- 📍 **Geospatial Analysis**: Distance-based fraud detection using Haversine formula
- ⏰ **Temporal Pattern Recognition**: Time-based fraud pattern analysis
- 🎯 **High Accuracy**: Optimized model with SMOTE for handling imbalanced data
- 🌐 **Web Interface**: Beautiful Streamlit-based user interface
- 📊 **Interactive Dashboard**: Real-time predictions with detailed insights

## 🏗️ Architecture

```
Credit Card Fraud Detection System
├── 📁 Data Processing
│   ├── Feature Engineering
│   ├── Geospatial Calculations
│   └── Categorical Encoding
├── 🤖 Machine Learning
│   ├── LightGBM Classifier
│   ├── SMOTE Balancing
│   └── Model Persistence
└── 🌐 Web Application
    ├── Streamlit Interface
    ├── Real-time Predictions
    └── User-friendly Forms
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Theophilus18-cyber/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv fraud_detection_env
   
   # On Windows
   fraud_detection_env\Scripts\activate
   
   # On macOS/Linux
   source fraud_detection_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to access the fraud detection interface.

## 📊 Model Performance

The LightGBM model achieves excellent performance metrics:

- **Accuracy**: High classification accuracy
- **Precision**: Low false positive rate
- **Recall**: High fraud detection rate
- **ROC-AUC**: Excellent discriminative ability

### Key Features Used:
- `merchant`: Merchant identifier
- `category`: Transaction category
- `amt`: Transaction amount
- `distance`: Calculated distance between customer and merchant
- `hour`, `day`, `month`: Temporal features
- `gender`: Customer demographic
- `cc_num`: Credit card number (hashed)

## 🎮 Usage

### Web Interface

1. **Enter Transaction Details**:
   - Merchant name and category
   - Transaction amount
   - Customer and merchant coordinates
   - Transaction timing
   - Customer information

2. **Get Instant Results**:
   - Real-time fraud prediction
   - Confidence scores
   - Detailed analysis

### API Usage

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoders.jb")

# Prepare input data
input_data = pd.DataFrame([[...]])  # Your transaction data

# Make prediction
prediction = model.predict(input_data)
```

## 🔧 Technical Details

### Data Processing Pipeline

1. **Feature Engineering**:
   - Temporal feature extraction from timestamps
   - Geospatial distance calculation using Haversine formula
   - Categorical encoding for merchant, category, and gender

2. **Data Balancing**:
   - SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance
   - Ensures robust model training on both fraud and legitimate transactions

3. **Model Training**:
   - LightGBM with optimized hyperparameters
   - Cross-validation for model validation
   - Feature importance analysis

### Key Algorithms

- **LightGBM**: Gradient boosting framework for high-performance classification
- **SMOTE**: Synthetic data generation for balanced training
- **Haversine Formula**: Accurate geospatial distance calculation
- **Label Encoding**: Efficient categorical variable processing

## 📁 Project Structure

```
credit-card-fraud-detection/
├── 📄 app.py                 # Streamlit web application
├── 📓 app.ipynb             # Jupyter notebook with model development
├── 📊 dataset.csv           # Training dataset
├── 🤖 fraud_detection_model.jb  # Trained LightGBM model
├── 🔧 label_encoders.jb     # Label encoders for categorical variables
├── 📋 requirements.txt      # Python dependencies
├── 🚫 .gitignore           # Git ignore rules
└── 📖 README.md            # Project documentation
```

## 🛠️ Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **LightGBM**: Gradient boosting machine learning
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Geopy**: Geospatial calculations

### Additional Libraries
- **Matplotlib & Seaborn**: Data visualization
- **Imbalanced-learn**: Handling class imbalance
- **Joblib**: Model persistence

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Theophilus Kgopa**
- GitHub: [@Theophilus18-cyber](https://github.com/Theophilus18-cyber)
- LinkedIn: [Theophilus Kgopa](https://linkedin.com/in/theophilus-kgopa)

## 🙏 Acknowledgments

- Credit card fraud detection dataset providers
- LightGBM development team
- Streamlit community
- Open-source machine learning community

## 📞 Support

If you have any questions or need support, please:

1. Check the [Issues](https://github.com/Theophilus18-cyber/credit-card-fraud-detection/issues) page
2. Create a new issue if your problem isn't already addressed
3. Contact the author directly

---

<div align="center">
  <p>Made with ❤️ for secure financial transactions</p>
  <p>⭐ Star this repository if you find it helpful!</p>
</div> 