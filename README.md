# ChurnShield

**ChurnShield** is an AI-driven web application designed to predict customer churn and recommend personalized retention strategies. Built using advanced machine learning and a Flask web framework, ChurnShield empowers businesses to reduce customer attrition through predictive analytics and targeted retention measures.

---

## Project Overview

In competitive markets, **customer churn** can severely impact a business’s growth and sustainability. **ChurnShield** addresses this issue by providing:

- Accurate churn predictions using a trained machine learning model.
- Actionable retention strategies based on customer data.
- A user-friendly admin dashboard to monitor and manage churn risks.

---

## Problem Statement

Businesses, especially those with subscription-based models, face two major challenges:

-  **High Churn Rates**: Leading to revenue loss and reduced lifetime value.
-  **Difficulty Identifying At-Risk Customers**: No early warning systems to intervene effectively.

---

## Objectives

- Develop a churn prediction model using behavioral and transactional data.
- Perform exploratory data analysis (EDA) with actionable insights.
- Engineer key features (e.g., Recency, Usage Intensity, Support Call Frequency).
- Integrate the model into a Flask web app for real-time prediction.
- Build an admin dashboard with user management, prediction history, and EDA visualizations.

---

##  Technologies Used

###  Backend
- **Python 3.10.6**
- **Flask 2.3.3**
- **Flask-SQLAlchemy 3.0.3** (ORM)
- **Flask-Login 0.6.3** (Authentication)
- **Flask-Migrate 4.1.0** (Migrations)

###  Machine Learning
- **Scikit-learn 1.5.1** (Random Forest Classifier)
- **Pandas 2.0.3**, **Numpy 1.24.4**
- **Joblib 1.4.2** (Model serialization)

###  Frontend
- **HTML5, CSS3, Bootstrap**
- **Jinja2 3.1.6** (Template rendering)

###  Other Tools
- **Gunicorn 23.0.0** (Production WSGI server)
- **MarkupSafe, Click, Greenlet, Werkzeug** (Support libraries)

---

##  System Architecture

ChurnShield follows a modular MVC-like architecture:

###  Modules:

1. **User Interface**  
   - Input form for customer data  
   - Real-time prediction and strategy output  
   - Admin dashboard with filtered prediction history and EDA access

2. **Authentication & Role Management**  
   - Role-based access (admin/users)  
   - Secure login via Flask-Login  
   - Password hashing via Werkzeug

3. **Customer Prediction Engine**  
   - Random Forest model via `scikit-learn`  
   - Predicts churn probability using:
     ```python
     churn_prob = best_model_pipeline.predict_proba(customer_data)[:, 1][0]
     ```

4. **Feature Engineering**  
   - Custom features include:
     | Feature | Formula |
     |--------|---------|
     | Monthly Spend | Total Spend ÷ Tenure |
     | Recency | max(Last Interaction) – Last Interaction |
     | Support Call Frequency | Support Calls ÷ Tenure |
     | Payment Reliability | 1 ÷ (Payment Delay + 1) |
     | Subscription Spend | Monthly Spend × Subscription Type |
     | Tenure Contract | Tenure × Contract Length |

5. **Retention Strategy Generator**  
   - Activated if churn probability > 0.5  
   - Generates strategies based on behavioral features  
   - Sorted by feature importance

6. **Data Storage Module**  
   - Managed via **Flask-SQLAlchemy** and **SQLite**  
   - Tables:
     - `User`: id, username, password, is_admin
     - `Prediction`: id, prediction, input_data, retention_strategy, user_id, date_submitted

7. **Admin Dashboard & CSV Export**  
   - Filter predictions by user/date  
   - View user activity and export prediction logs as CSV

8. **EDA & Visualization**  
   - Pre-generated EDA charts hosted in `/static/images`  
   - Visual insights include:
     - Churn Distribution
     - Support Calls & Payment Delay (Boxplots)
     - Age Distribution by churn
     - Correlation Matrix

---

## Sample EDA Graphs

| Graph | Description |
|-------|-------------|
| **Churn Distribution** | Highlights class imbalance |
| **Feature Boxplots** | Shows how features like Support Calls vary by churn status |
| **Age Distribution** | Visualizes customer age vs churn likelihood |
| **Correlation Matrix** | Correlations between features and churn |

---

