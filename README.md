# 🧠 Neuroadaptive EEG Emotion Detection System

## Overview
A machine learning-based system that predicts emotional states (Happy, Sad, Angry, Calm) from EEG brainwave signals. The project uses frequency band power features extracted from EEG data to classify emotions using advanced ML algorithms.

## 🎯 Features
- **EEG Signal Processing**: Extracts power features from 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Emotion Classification**: Predicts 4 emotional states based on Valence-Arousal model
- **FastAPI Backend**: RESTful API for real-time predictions
- **Web Interface**: User-friendly HTML interface for emotion prediction
- **Model Registry**: Version control and management for trained models
- **Automated Pipeline**: End-to-end training pipeline from data ingestion to evaluation

## 📊 Emotion Classification Model
The system uses the **Valence-Arousal** dimensional model:
- **Happy**: High Valence (≥5) + High Arousal (≥5)
- **Calm**: High Valence (≥5) + Low Arousal (<5)
- **Angry**: Low Valence (<5) + High Arousal (≥5)
- **Sad**: Low Valence (<5) + Low Arousal (<5)

## 🧬 EEG Frequency Bands
| Band | Frequency Range | Associated State |
|------|----------------|------------------|
| Delta (δ) | 0.5-4 Hz | Deep sleep, unconscious |
| Theta (θ) | 4-8 Hz | Drowsiness, meditation |
| Alpha (α) | 8-13 Hz | Relaxation, calmness |
| Beta (β) | 13-30 Hz | Active thinking, focus |
| Gamma (γ) | 30-45 Hz | High-level cognition |

## 🏗️ Project Structure
```
Neuroadaptive_ML/
├── main.py                    # FastAPI application
├── run_pipeline.py            # Training pipeline orchestrator
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── README.md                  # Project documentation
│
├── templates/                 # HTML templates
│   ├── index.html            # Input form page
│   └── result.html           # Prediction result page
│
├── src/
│   ├── components/           # Core ML components
│   │   ├── data_ingestion.py        # Load EEG .dat files
│   │   ├── data_transformation.py   # Signal processing & feature extraction
│   │   ├── feature_engineering.py   # Label encoding & train-test split
│   │   ├── model_trainer_lite.py    # Model training
│   │   ├── model_evaluation.py      # Performance metrics
│   │   └── model_registry.py        # Model versioning
│   │
│   ├── pipeline/             # ML pipelines
│   │   ├── train_pipeline.py        # Training workflow
│   │   └── predict_pipeline.py      # Inference workflow
│   │
│   ├── logger.py             # Logging utility
│   ├── exception.py          # Custom exception handling
│   └── utils.py              # Helper functions
│
├── data/                     # Data storage
│   ├── raw/                  # Raw EEG .dat files
│   ├── ingested/             # Loaded numpy arrays
│   ├── processed/            # Extracted features
│   └── final/                # Train-test splits
│
├── artifacts/                # Model artifacts
│   ├── model_registry/       # Versioned models
│   ├── evaluation/           # Performance reports
│   ├── best_model.pkl        # Best trained model
│   └── preprocessor.pkl      # Feature preprocessor
│
└── logs/                     # Execution logs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Neuroadaptive_ML

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📚 Usage

### 1. Train the Model
Run the complete training pipeline:
```bash
python run_pipeline.py
```

This executes:
1. **Data Ingestion**: Loads EEG .dat files from `data/raw/`
2. **Data Transformation**: Extracts frequency band features using FFT
3. **Feature Engineering**: Creates emotion labels and splits data
4. **Model Training**: Trains XGBoost classifier with hyperparameter tuning
5. **Model Registry**: Saves versioned model artifacts

### 2. Start the Web Application
```bash
python main.py
```

The server starts at `http://localhost:8000`

### 3. Make Predictions
- Open browser and navigate to `http://localhost:8000`
- Enter EEG band power values (normalized 0-1)
- Click "Predict Emotion" to get results

### 4. API Endpoints

#### Home Page
```
GET /
Returns: HTML form for EEG input
```

#### Predict Emotion
```
POST /predict
Form Data:
  - delta: float (Delta band power)
  - theta: float (Theta band power)
  - alpha: float (Alpha band power)
  - beta: float (Beta band power)
  - gamma: float (Gamma band power)

Returns: HTML page with predicted emotion and confidence
```

#### Health Check
```
GET /health
Returns: {"status": "healthy", "model_loaded": true}
```

## 🔬 Technical Details

### Data Processing Pipeline
1. **Signal Loading**: Reads DEAP dataset .dat files (pickle format)
2. **Feature Extraction**: Computes mean power in 5 frequency bands using FFT
3. **Normalization**: Applies StandardScaler for feature scaling
4. **Label Creation**: Maps Valence-Arousal ratings to emotion categories

### Machine Learning Model
- **Algorithm**: XGBoost Classifier
- **Features**: 5 frequency band power values
- **Classes**: 4 emotions (Angry, Calm, Happy, Sad)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Hyperparameter Tuning**: Grid search with cross-validation

### Model Registry
Versioned model storage with metadata:
- Model version ID
- Training timestamp
- Performance metrics
- File paths for model and preprocessor

## 📈 Performance
The model achieves:
- **Accuracy**: ~85-90% (varies by dataset)
- **Training Time**: ~2-5 minutes on standard hardware
- **Inference Time**: <100ms per prediction

## 🛠️ Development

### Adding New Features
1. Modify `data_transformation.py` to extract additional features
2. Update `predict_pipeline.py` input validation
3. Retrain model using `run_pipeline.py`

### Custom Model Training
Edit `model_trainer_lite.py` to:
- Add new algorithms
- Modify hyperparameters
- Change evaluation metrics

## 📝 Data Format

### Input EEG Data (.dat files)
```python
{
    'data': numpy.ndarray,    # Shape: (trials, channels, samples)
    'labels': numpy.ndarray   # Shape: (trials, 4) - [Valence, Arousal, Dominance, Liking]
}
```

### Prediction Input
```json
{
    "delta": 0.25,
    "theta": 0.30,
    "alpha": 0.35,
    "beta": 0.40,
    "gamma": 0.45
}
```

### Prediction Output
```json
{
    "predicted_emotion": "Happy",
    "confidence_score": 0.8542,
    "model_version": "EEG_MODEL_v1_20260219_130306"
}
```

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure you've run `python run_pipeline.py` first
- Check `artifacts/model_registry/` for saved models

### Import Errors
- Verify all dependencies: `pip install -r requirements.txt`
- Activate virtual environment

### Data Loading Issues
- Place EEG .dat files in `data/raw/` directory
- Ensure files follow DEAP dataset format

## 🔮 Future Enhancements
- [ ] Real-time EEG streaming support
- [ ] Multi-modal emotion detection (EEG + facial expressions)
- [ ] Deep learning models (CNN, LSTM)
- [ ] Mobile application
- [ ] Cloud deployment (AWS/Azure)
- [ ] User authentication and history tracking

## 📄 License
MIT License - See LICENSE file for details

## 👥 Contributors
Developed for Neuroadaptive EdTech Hackathon

## 📧 Contact
For questions or support, please open an issue on GitHub.

---
**Note**: This system is for research and educational purposes. Not intended for clinical diagnosis.