# ChurnShield - Customer Churn Prediction Web App

ChurnShield is an AI-powered web application that predicts customer churn and provides personalized retention strategies. 

Built with Flask and integrated with a machine learning pipeline, the app enables businesses to make data-driven decisions for improving customer retention.

<img width="60" height="60" alt="logo" src="https://github.com/user-attachments/assets/f2d111d9-b2d3-4d6c-bbbb-1a5515a10d74" />

---

## Motivation

In today’s data-driven landscape, customer churn directly affects profitability and long-term growth. ChurnShield solves this by:

- Predicting which customers are at risk of leaving.
- Providing actionable strategies tailored to each customer.
- Offering an admin dashboard to manage users, predictions, and insights.

---

## 🛠Tech Stack

| Layer        | Technologies |
|--------------|--------------|
| **Frontend** | HTML5, CSS3, Bootstrap, Jinja2 |
| **Backend**  | Flask, Flask-SQLAlchemy, Flask-Login |
| **ML Model** | Scikit-learn (Random Forest), Joblib |
| **Database** | SQLite + Flask-Migrate |
| **Deployment** | Railway, Gunicorn, Nixpacks |
| **Other**    | Pandas, NumPy, CSV Export, EDA Visualizations |

---

## 📁 Project Structure

```
ChurnShield-App/
│
├── app.py                                                         # Main Flask app
├── app copy.py                                                    # Backup / alternate version
├── database.py                                                    # DB setup and config
├── inspect_model.py                                               # Model inspection / testing
├── requirements.txt                                               # Python dependencies
├── runtime.txt                                                    # Python runtime for deployment (Railway)
├── nixpacks.toml                                                  # Nixpacks config (buildpacks)
│
├── data.csv                                                       # Sample dataset
├── dirlist.txt                                                    # Utility: directory listing
├── minorupgrades.txt                                              # Notes for improvements
├── Untitled design.png                                            # Project banner/logo
│
├── migrations/                                                    # Flask-Migrate folder
├── instance /                                                     # Config & local SQLite DB
├── models/                                                        # ML model
├── static/                                                        # CSS, JS, images, EDA graphs
├── templates/                                                     # HTML templates (base.html, etc.)
│
└── .github/workflows/                                             # GitHub Actions CI/CD
└── deploy.yml                                                     # Deploy workflow
```
---

## Features

- 🎯 **Churn Prediction**: Trained Random Forest Classifier
- 📊 **EDA & Insights**: Pre-generated visualizations (boxplots, correlation, etc.)
- 🧠 **Feature Engineering**: Recency, Support Calls, Spend, Tenure, etc.
- 🧾 **Retention Strategy Generator**: Dynamic suggestions based on feature weights
- 🧑‍💻 **User Authentication**: Admin/user roles with Flask-Login
- 📂 **Admin Dashboard**:
  - Manage users & predictions
  - View EDA charts
  - Export prediction history as CSV

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bijay-odyssey/ChurnShield-App.git
cd ChurnShield-App
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up the Database
```bash
flask db init
flask db migrate
flask db upgrade
```

### 5. Run the Application
```bash
python app.py
```

Visit the app at: http://localhost:5000
---

## Deployment (Railway or any WSGI server)

The project is pre-configured for deployment via Railway using:

- runtime.txt → Specifies Python version
- nixpacks.toml → Used by Railway for build configuration
- deploy.yml → GitHub Actions deployment workflow
- Gunicorn → Production WSGI server
---

 

 
