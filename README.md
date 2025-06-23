# Fashion Product Image Classifier

## Overview
This project explores the use of **deep learning** for automated product categorization in e-commerce. The goal is to accurately classify product images into **Apparel** or **Footwear**, streamlining the tagging process, improving user experience, and enhancing recommendation systems.

Prepared by: Vy Nguyen, Becky Wang, Chia-Chien Chang, Xinying Wu, Yunling Li

Key results include:
- Final neural network (30-4 hidden layers) achieved **99.83% validation and test accuracy**
- Outperformed logistic regression by **0.53%**, which translates to significant cost and labor savings at scale
- Hyperparameter tuning and regularization techniques contributed to strong generalization and low overfitting
- Business analysis estimated this improvement could reduce misclassification for over 10,000 products daily, minimizing manual corrections and customer service load

> **Strategic Insight:** Even a small gain in model accuracy can translate into major business value in high-volume platforms — enabling scalable product tagging, better recommendations, and improved customer trust.

---

## Dataset
- **Source:** [Fashion Product Image Dataset on Kaggle](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images/data)  
- **Scope:** 2,906 labeled images (Apparel = 46%, Footwear = 54%)  
- **Image Size:** Varies from 1080×1440 to 1800×2400  
- **Target Variable:** Product Category (0 = Apparel, 1 = Footwear)

### 📋 Data Dictionary

| Feature         | Description                                      | Type       |
|-----------------|--------------------------------------------------|------------|
| Product Image   | Raw product photo (input to neural network)      | Image      |
| Category        | Apparel or Footwear (binary target variable)     | Categorical|

**Data Split:**
- Training: 1,743 images  
- Validation: 581 images  
- Test: 582 images  
(Stratified 60/20/20 split to preserve class balance)

---

## Tools & Methodology Overview
**Languages and Libraries:** Python (TensorFlow/Keras), NumPy, Matplotlib

### Selected Model Architecture:
- Input → Hidden Layer 1 (30 neurons, ReLU) → Hidden Layer 2 (4 neurons, ReLU) → Output (Sigmoid)

### Training Process:
- Optimizer: SGD with momentum = 0.5  
- Learning Rate: 0.001  
- Weight Initialization: He Normal  
- Epochs: 200  
- Regularization: L2 penalty (λ = 0.001) and Dropout (rate = 0.4)

### Experimental Tuning:
- Explored various hidden layers (1 vs. 2), neuron sizes, and learning rates
- Compared optimizers: Adam, RMSprop, SGD, and Momentum
- Tested regularization methods: L2, Dropout, Batch Normalization
- Benchmarked alternative architectures: CNN, MobileNetV2, RNN-GRU (no added benefit)

---

## Highlighted Visualizations

**Loss Curve with He Normal Initialization:**  
`Notebooks/Loss Curve of He Normal Initialization.png`

**Effect of Regularization (L2 + Dropout):**  
`Notebooks/Loss Curve of L2 and Dropout Regularization.png`

**Cross-Validation Accuracy by Hidden Layers & Alpha:**  
`Notebooks/CV Accuracy by Alpha and Hidden Layer Size.png`

---

## Results & Key Insights

### Final Model Performance

| Model                | Train Accuracy | Validation Accuracy | Test Accuracy |
|----------------------|----------------|----------------------|---------------|
| Logistic Regression  | 99.7%          | 99.3%                | 99.3%         |
| Neural Net (7-4)     | 99.5%          | 99.83%               | 99.83%        |
| Neural Net (30-4)    | 99.6%          | **99.83%**               | **99.83%**    |

- Best performing architecture was a DNN with (30, 4) hidden units
- No overfitting observed — validation loss tracked training loss closely
- Hyperparameter tuning (especially with regularization) stabilized learning and ensured generalization

---

## Key Deliverables
- Python code: `Notebooks/Final Code_Team A4.ipynb`
- Final project report: `Reports/Project Report_Team_A4.pdf`

---

## What I Learned
- Even simple DNNs can outperform logistic regression when tuned well, especially for structured binary tasks
- Weight initialization, regularization, and optimizer choice play a crucial role in training stability
- Small accuracy improvements can drive big downstream value in real-world, high-volume applications
- Experimental tracking across architectures is key to building confidence in model choice

---

## What I Plan to Improve
- Explore convolutional neural networks with more aggressive data augmentation to scale across more categories
- Test deployment pipelines for automated image tagging in real-time
- Evaluate ensemble or stacking methods to combine logistic and deep models
- Expand dataset with multi-class product categories to improve platform relevance

---

## About Me
Hi, I’m Vy Nguyen and I’m currently pursuing my MS in Business Analytics at UC Irvine. I’m passionate about data analytics in Finance and Retail Applications. Connect with me on [LinkedIn](https://www.linkedin.com/in/vy-ngoc-lan-nguyen).

