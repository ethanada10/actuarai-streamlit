# 🛡️ ActuarAI — Insurance Fraud Detection App

> A production-ready **AI-powered web application** for detecting potential insurance fraud using machine learning, containerized with Docker and deployed on the cloud.

🌍 Live Demo: https://actuarai-streamlit.onrender.com

---

## 🚀 Overview

**ActuarAI** is an end-to-end machine learning project that:

* Trains a fraud detection model on structured insurance data
* Serves predictions through an interactive **Streamlit web interface**
* Runs inside a **Docker container** for reproducibility
* Is deployed as a **public cloud service** via Render

This project demonstrates real-world **ML engineering, deployment, and MLOps fundamentals** — from data processing and model training to production deployment.

---

## 🧠 Key Features

* 📊 Interactive Streamlit dashboard
* 🤖 Machine Learning fraud classification model
* 📁 CSV data ingestion
* 🔍 Feature validation with expected schema
* 💾 Model persistence using `joblib`
* 🐳 Fully Dockerized
* ☁️ Cloud deployment (Render)

---

## 🏗️ Architecture

```
User → Web Browser
        ↓
   Streamlit UI
        ↓
 Fraud ML Model
        ↓
 Prediction Output

Docker Container
        ↓
 Render Cloud Platform
```

---

## 📂 Project Structure

```
ActuarAI_streamlit/
├── app.py                # Streamlit web app
├── train.py             # Model training pipeline
├── utils.py             # Helper functions
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build instructions
├── render.yaml         # Render deployment config
├── data/
│   └── insurance_fraud_dataset.csv
├── models/
│   ├── fraud_model.joblib
│   └── expected_columns.joblib
└── README.md
```

---

## ⚙️ Tech Stack

| Category         | Technology    |
| ---------------- | ------------- |
| Language         | Python 3.11   |
| ML               | Scikit-learn  |
| Frontend         | Streamlit     |
| Containerization | Docker        |
| Deployment       | Render        |
| Data             | Pandas, NumPy |

---

## 🧪 Machine Learning Pipeline

1. **Data Loading**

   * Reads structured insurance data from CSV

2. **Preprocessing**

   * Feature validation using expected schema
   * Cleaning and formatting

3. **Model Training**

   * Supervised classification model
   * Trained using Scikit-learn

4. **Model Persistence**

   * Saved using `joblib`

5. **Inference**

   * Model loaded inside Streamlit app
   * Real-time prediction via UI inputs

---

## 🐳 Run Locally with Docker

### 1️⃣ Build Image

```bash
docker build -t actuarai .
```

### 2️⃣ Run Container

```bash
docker run -p 8501:8501 actuarai
```

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## ☁️ Cloud Deployment

This application is deployed using:

* **Docker** for environment consistency
* **Render Web Services** for hosting

The service automatically builds and runs the container from the GitHub repository.

---

## 🎯 Use Cases

* Insurance fraud detection demo
* Machine learning deployment portfolio
* MLOps and cloud engineering showcase
* Interactive data science application

---

## 📈 Future Improvements

* 🔐 User authentication
* 📊 Model performance dashboard
* 🧠 Deep learning-based classifier
* 🗃️ Database integration
* 📦 CI/CD pipeline (GitHub Actions)

---

## 👨‍💻 Author

**Ethan Ada**
Master’s Student in Mathematical Engineering
Data Scientist / Quant & ML Engineering Track




---

## ⭐ Why This Project Matters

This project demonstrates:

* Real-world ML deployment
* Production containerization
* Cloud hosting
* End-to-end system design

It goes beyond notebooks and shows how to turn **AI models into real services**.

---

If you like this project, feel free to ⭐ the repo and connect!
