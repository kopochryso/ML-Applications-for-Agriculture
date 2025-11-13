🌾 End-to-End ML for Smart Beetroot Cultivation

Yield Prediction • Stress Detection • Fertilizer Optimization • Irrigation Scheduling • Quality Prediction

This repository implements a complete, end-to-end Machine Learning workflow using an open beetroot cultivation dataset
(Kenter & Hoffman, 2025 – https://odjar.org/article/view/18784/18278
).



The project demonstrates how ML can support agronomic decision-making across the growing season, including:
🔮 Yield Prediction
🌡️ Weather-driven stress modeling
🌱 Nitrogen/Fertilizer Optimization
💧 Irrigation Scheduling Recommendation
🧪 Crop Quality Prediction
📊 Agronomic Feature Engineering
It also includes MLOps components, such as data pipelines, model versioning, experiment tracking, and deployable inference services.



📁 Repository Structure
beetroot-ml-platform/   
│   
├── data/      
│   ├── raw/                       # Original dataset
│   ├── processed/              # Cleaned + engineered data
│   
├── notebooks/  
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── FeatureEngineering.ipynb    
│   ├── ModelTraining.ipynb 
│   
├── src/    
│   ├── data_pipeline.py        # Automatic preprocessing pipeline
│   ├── train_model.py        # Training script (CLI)
│   ├── evaluate.py          # Evaluation + SHAP explainability
│   ├── inference_api.py        # FastAPI microservice for deployment
│   
├── models/ 
│   ├── model_latest.pkl    
│   ├── model_v1/              # Versioned models
│   
├── mlflow/                  # MLflow experiment logs
├── docker/ 
│   ├── Dockerfile  
│   
├── tests/  
│   ├── test_preprocessing.py   
│   ├── test_inference.py   
│   
├── requirements.txt    
└── README.md   



🚀 How to Run the Pipeline
1. Install dependencies
pip install -r requirements.txt

2. Preprocess data
python src/data_pipeline.py --input data/raw --output data/processed

3. Train models
python src/train_model.py --config configs/yield_config.yaml

4. Serve the trained model via API
uvicorn src.inference_api:app --reload



📊 Model Performance (examples)
Yield prediction RMSE (g/plant), Stress classifier accuracy (%), Ferilizer optimization model MAE (kg/ha)



🔧 MLOps Components Included
✔ MLflow experiment tracking
✔ Versioned data pipelines
✔ Reproducible environment (requirements.txt + Dockerfile)
✔ Unit tests
✔ Modular training scripts
✔ FastAPI inference server
✔ Config-driven training (YAML)
✔ Feature engineering pipeline