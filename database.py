from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin  # Import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):  # Add UserMixin
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # Hashed in app.py
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

 
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.Float, nullable=False)  # Probability between 0 and 1
    date_submitted = db.Column(db.DateTime, nullable=False)
    input_data = db.Column(db.String(500), nullable=False)  # Store input as JSON
    retention_strategy = db.Column(db.String(500), nullable=True)  # New column



def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///churn_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()