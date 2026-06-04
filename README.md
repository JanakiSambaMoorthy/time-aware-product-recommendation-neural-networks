# Time-Aware Sequential Neural Networks for Next-Best Product Recommendation

## Overview
This project builds a time-aware next-best product recommendation system using the Instacart Market Basket dataset. The goal is to predict the next product a customer is likely to purchase based on their historical shopping sequence and temporal purchase behavior.

The project compares recurrent neural network architectures and a Transformer-based model for sequential recommendation.

## Business Problem
Online grocery and e-commerce platforms need to personalize product recommendations based on customer purchase patterns. A simple popularity-based recommendation system recommends the same products to many users, but customer behavior is often sequential and time-dependent.

This project answers the question:

**Can customer purchase history, product hierarchy, and temporal shopping patterns be used to predict the next product a customer is likely to buy?**

## Project Objectives
- Perform data understanding and preprocessing on the Instacart dataset.
- Build user-level chronological purchase sequences.
- Engineer time-aware and product hierarchy features.
- Train multiple neural network recommendation models.
- Compare LSTM, GRU, hybrid activation, and Transformer-based architectures.
- Evaluate models using classification metrics and Top-K recommendation metrics.
- Identify the best-performing model for next-best product recommendation.

## Dataset
Dataset used: Instacart Market Basket dataset.

Expected files:
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

The dataset is not included in this repository. Place the files inside the `data/` folder before running the notebooks.

## Models Implemented
- Popularity baseline
- LSTM + ReLU + Attention
- GRU + ReLU + Attention
- GRU + Swish / SwiLU variant
- GRU + ReLU-GELU hybrid activation variant
- LSTM + ReLU-GELU hybrid activation variant
- Transformer Encoder + GELU

## Evaluation Metrics
The project evaluates performance using both classification and recommendation metrics:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Precision@5
- Precision@10
- Recall@5
- Recall@10
- Hit Rate@5
- Hit Rate@10

## Key Results
The recurrent attention-based models produced the strongest recommendation performance.

The best model was:

**LSTM + ReLU + Attention**

It achieved the strongest overall Top-K recommendation performance, with the highest reported Recall@10 of approximately **0.1469**.

The GRU + ReLU + Attention model performed very closely, with reported Recall@10 of approximately **0.1459**.

## Repository Structure
```text
time-aware-product-recommendation-neural-networks/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_understanding_and_preprocessing.ipynb
│   ├── 02_sequence_construction_and_batch_preparation.ipynb
│   ├── 03_lstm_relu_attention_model.ipynb
│   ├── 04_gru_swish_attention_model.ipynb
│   ├── 05_gru_relu_gelu_attention_model.ipynb
│   ├── 06_lstm_relu_gelu_attention_model.ipynb
│   ├── 07_transformer_encoder_gelu_model.ipynb
│   └── 08_consolidated_results_and_model_comparison.ipynb
│
├── data/
│   └── README.md
│
├── reports/
│   └── README.md
│
├── visuals/
│   └── README.md
│
└── src/
    └── README.md
```

## How to Run
1. Download the Instacart Market Basket dataset.
2. Place the dataset CSV files inside the `data/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the notebooks in numerical order from `01` to `08`.

## Notes
- The notebooks were originally developed in Google Colab.
- Some file paths may need to be updated before running locally.
- The final report is not included in this public-ready folder because the original file contains student-identifying information.
- A redacted report can be added later if needed.

## Author
Janaki S  
MS Data Science, University of Memphis  
Deep Learning | Recommendation Systems | Retail Analytics | Data Science
