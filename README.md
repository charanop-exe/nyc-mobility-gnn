# 🚕 Spatio-Temporal Taxi Demand Forecasting using Graph Neural Networks

A production-ready spatio-temporal machine learning system for **zone-level NYC taxi demand prediction**, integrating **graph learning, temporal modeling, and real-time deployment**.

---

## 📌 Project Motivation

Urban mobility systems exhibit strong **spatial and temporal dependencies**.  
Taxi demand in one region is influenced not only by its historical trends but also by **neighboring regions and time-of-day effects**.

Traditional time-series or regression models fail to capture:
- spatial interdependencies between regions  
- non-Euclidean graph structures  
- dynamic temporal evolution  

This project addresses these challenges by modeling **NYC taxi zones as a graph** and applying a **Spatio-Temporal Graph Neural Network (ST-GNN)** to forecast hourly demand.

---

## 🧠 Key Contributions

- Graph-based spatial modeling using NYC taxi zone adjacency
- Temporal feature engineering with sliding windows
- Deep learning using **Graph Neural Networks**
- Robust evaluation with regression metrics
- **Real-time inference deployment** using FastAPI
- **Interactive visualization** via Streamlit

---

## 🗂 Dataset Description

### 1. NYC Yellow Taxi Trip Records
- Source: NYC Taxi & Limousine Commission (TLC)
- Format: Parquet
- Time resolution: Hourly aggregation
- Target variable: Number of pickups per zone per hour

### 2. NYC Taxi Zones Shapefile
- Defines **263 official NYC taxi zones**
- Used to derive:
  - spatial adjacency graph
  - zone identifiers and names
  - geographic consistency

After preprocessing and filtering inactive zones, approximately **261 active zones** were retained.

---

## 🗺 Graph Construction (Spatial Modeling)

Each taxi zone is treated as a **node** in a graph.

Edges are created when two zones **share a physical boundary**, computed using geometric adjacency from the official taxi zone shapefile.

This produces:
- an undirected spatial graph  
- non-uniform neighborhood sizes  
- realistic urban connectivity  

The graph is represented using an **edge index** format compatible with PyTorch Geometric.

---

## ⏱ Temporal Modeling Strategy

### Sliding Window Formulation

A fixed-length look-back window of **T hours** is used to predict the next hour.

Input features include:
- historical demand values  
- normalized hour-of-day  
- normalized day-of-week  

Final input tensor shape:

samples × time_window × number_of_zones × number_of_features

---

## 🧠 Model Architecture (Technical Core)

### Why Graph Neural Networks?

NYC taxi zones form an **irregular spatial structure**.  
Grid-based models (e.g., CNNs) assume uniform spacing and are unsuitable.

Graph Neural Networks enable:
- message passing between neighboring zones  
- learning spatial spillover effects  
- scalability to non-Euclidean urban layouts  

---

### Architecture Overview

**Spatial Encoder**
- Graph convolution / graph attention layers
- Aggregates neighborhood information
- Captures spatial dependencies

**Temporal Encoder**
- Sliding-window temporal learning
- Models short-term demand dynamics
- Learns periodic patterns

**Prediction Head**
- Fully connected layers
- Outputs next-hour demand per zone

The design balances **expressiveness** and **stability** to avoid overfitting sparse zones.

---

## 📊 Evaluation Methodology

This is a **regression task**.

### Metrics Used

| Metric | Description |
|------|------------|
| RMSE | Penalizes large prediction errors |
| MAE  | Interpretable average absolute error |
| R²   | Proportion of explained variance |

These metrics assess accuracy, robustness, and generalization.

---

## 📈 Results Summary

- High-demand hubs (airports, downtown areas) are consistently identified
- Low-activity residential zones remain stable
- Spatial modeling improves performance over non-graph baselines

The results demonstrate that **explicit spatial dependency modeling enhances demand forecasting accuracy**.

---

## 🌐 Deployment (Production Readiness)

### FastAPI Backend
- Loads trained ST-GNN model
- Exposes a `/predict` REST endpoint
- Returns zone-level demand forecasts as JSON

### Streamlit Dashboard
- Interactive hour-based prediction
- Human-readable NYC taxi zone names
- Clean, interpretable visualizations
- Designed for non-technical users

---

## 📸 Screenshots

Add screenshots inside a `screenshots/` folder and reference them here:

- Streamlit Dashboard  
- Top Demand Zones Visualization  
- FastAPI Prediction Output  

(Example filenames: `dashboard.png`, `top_zones.png`, `api_output.png`)

---

## 📁 Project Structure

spatial_temporal_mobility  
├── src  
│   ├── model.py  
│   ├── train.py  
│   ├── evaluate.py  
│   ├── api.py  
│   └── app.py  
│  
├── data  
│   ├── raw  
│   │   └── taxi_zones.shp  
│   └── processed  
│       ├── final_dataset.npz  
│       └── model_weights.pth  
│  
└── README.md  

---

## 🔬 Limitations & Future Work

- Incorporation of weather and event data
- Multi-step forecasting (e.g., 24-hour horizon)
- Attention-based temporal encoders
- Real-time streaming inference
- Borough-level policy and planning analysis

---

## 🎓 Academic Relevance (MSDS Alignment)

This project demonstrates:
- advanced feature engineering  
- graph-based reasoning  
- applied deep learning  
- rigorous evaluation  
- end-to-end deployment  

It aligns strongly with MSDS coursework in:
- Machine Learning
- Deep Learning
- Data Engineering
- Urban Analytics
- Applied AI Systems

---

## 👤 Author

**Charan OP**  
Aspiring MSDS Student  
Focus Areas: Machine Learning, Graph Neural Networks, Urban Mobility Analytics

---

## 🏁 Final Note

This project is intentionally designed to be **technically rigorous, interpretable, reproducible, and deployable**.  
It reflects **graduate-level problem solving**, not just experimentation.
