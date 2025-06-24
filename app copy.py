from flask import Flask, render_template, request, redirect, url_for, flash, Response, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import json
from database import db, User, Prediction, init_db
from sqlalchemy import asc, desc
from flask_migrate import Migrate
import time
import csv
from io import StringIO


app = Flask(__name__)
app.secret_key = 'mykeyhere'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add custom Jinja filter for parsing JSON (moved after app definition)
app.jinja_env.filters['from_json'] = json.loads

# Initialize database
init_db(app)
migrate = Migrate(app, db)

# Load ML model and data with error handling
try:
    best_model = joblib.load('models/best_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    best_model = None

try:
    df = pd.read_csv('data.csv')
    churned = df[df['Churn'] == 1]
    non_churned = df[df['Churn'] == 0]
except Exception as e:
    print(f"Error loading data: {e}")
    df = None
    churned = None
    non_churned = None

# Feature mappings
contract_mapping = {'Monthly': 1, 'Quarterly': 2, 'Annual': 3}
subscription_mapping = {'Basic': 1, 'Standard': 2, 'Premium': 3}

# Define compute_thresholds function before calling it
def compute_thresholds(churned, non_churned):
    df['Monthly Spend'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
    df['Usage Intensity'] = df['Usage Frequency'] / df['Tenure'].replace(0, 1)
    df['Support Call Frequency'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
    df['Payment Reliability'] = 1 / (df['Payment Delay'] + 1)
    df['Contract Length Numeric'] = df['Contract Length'].map(contract_mapping)
    df['Tenure Contract'] = df['Tenure'] * df['Contract Length Numeric']
    df['Subscription Numeric'] = df['Subscription Type'].map(subscription_mapping)
    df['Subscription Spend'] = df['Monthly Spend'] * df['Subscription Numeric']
    df['Recency'] = df['Last Interaction'].max() - df['Last Interaction']

    churned = df[df['Churn'] == 1]
    non_churned = df[df['Churn'] == 0]

    thresholds = {
        'Support Calls': churned['Support Calls'].median(),
        'Total Spend': non_churned['Total Spend'].median(),
        'Payment Delay': churned['Payment Delay'].mean() + churned['Payment Delay'].std(),
        'Payment Reliability': non_churned['Payment Reliability'].median(),
        'Support Call Frequency': churned['Support Call Frequency'].quantile(0.75),
        'Recency': churned['Recency'].quantile(0.75),
        'Age': churned['Age'].quantile(0.75),
        'Monthly Spend': non_churned['Monthly Spend'].median(),
        'Tenure Contract': non_churned['Tenure Contract'].quantile(0.25),
        'Contract Length': non_churned['Contract Length Numeric'].median(),
        'Tenure': non_churned['Tenure'].quantile(0.25),
        'Subscription Spend': non_churned['Subscription Spend'].median(),
        'Usage Intensity': non_churned['Usage Intensity'].quantile(0.25),
        'Usage Frequency': non_churned['Usage Frequency'].quantile(0.25),
    }
    return thresholds

# Compute thresholds after the function is defined
thresholds = None
if df is not None:
    thresholds = compute_thresholds(churned, non_churned)
else:
    print("Thresholds not computed due to data loading error.")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
            return redirect(url_for('register'))
        user = User(username=username, password=generate_password_hash(password), is_admin=False)
        db.session.add(user)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash(f'Error registering user: {str(e)}')
            return redirect(url_for('register'))
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
        time.sleep(10)
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def dashboard():
#     if best_model is None or thresholds is None:
#         flash('Model or thresholds not loaded. Please contact the administrator.')
#         return redirect(url_for('dashboard'))

#     prediction = None
#     recommendations = []
#     if request.method == 'POST':
#         try:
#             # Validate and convert form inputs
#             data = {
#                 'Age': [float(request.form['age'])],
#                 'Gender': [request.form['gender']],
#                 'Tenure': [float(request.form['tenure'])],
#                 'Usage Frequency': [float(request.form['usage_frequency'])],
#                 'Support Calls': [float(request.form['support_calls'])],
#                 'Payment Delay': [float(request.form['payment_delay'])],
#                 'Subscription Type': [request.form['subscription_type']],
#                 'Contract Length': [request.form['contract_length']],
#                 'Total Spend': [float(request.form['total_spend'])],
#                 'Last Interaction': [float(request.form['last_interaction'])]
#             }

#             # Validate categorical inputs
#             if data['Gender'][0] not in ['Male', 'Female']:
#                 raise ValueError("Invalid gender value.")
#             if data['Subscription Type'][0] not in ['Basic', 'Standard', 'Premium']:
#                 raise ValueError("Invalid subscription type.")
#             if data['Contract Length'][0] not in ['Monthly', 'Quarterly', 'Annual']:
#                 raise ValueError("Invalid contract length.")

#             customer_data = pd.DataFrame(data)

#             customer_data['Recency'] = customer_data['Last Interaction'].max() - customer_data['Last Interaction']
#             customer_data['Monthly Spend'] = customer_data['Total Spend'] / customer_data['Tenure'].replace(0, 1)
#             customer_data['Usage Intensity'] = customer_data['Usage Frequency'] / customer_data['Tenure'].replace(0, 1)
#             customer_data['Support Call Frequency'] = customer_data['Support Calls'] / customer_data['Tenure'].replace(0, 1)
#             customer_data['Payment Reliability'] = 1 / (customer_data['Payment Delay'] + 1)
#             contract_numeric = customer_data['Contract Length'].map(contract_mapping)
#             customer_data['Tenure Contract'] = customer_data['Tenure'] * contract_numeric
#             subscription_numeric = customer_data['Subscription Type'].map(subscription_mapping)
#             customer_data['Subscription Spend'] = customer_data['Monthly Spend'] * subscription_numeric

#             churn_prob = best_model.predict_proba(customer_data)[:, 1][0]
#             prediction = round(churn_prob * 100, 2)
#             recommendations = recommend_retention_strategy(customer_data, churn_prob, thresholds)

#             pred = Prediction(
#                 user_id=current_user.id,
#                 prediction=churn_prob,
#                 date_submitted=pd.Timestamp.now(),
#                 input_data=json.dumps(data),
#                 retention_strategy=json.dumps(recommendations)
#             )
#             db.session.add(pred)
#             try:
#                 db.session.commit()
#             except Exception as e:
#                 db.session.rollback()
#                 flash(f'Error saving prediction: {str(e)}')
#                 return redirect(url_for('dashboard'))

#         except ValueError as ve:
#             flash(f'Invalid input: {str(ve)}')
#             return redirect(url_for('dashboard'))
#         except Exception as e:
#             flash(f'Error processing prediction: {str(e)}')
#             return redirect(url_for('dashboard'))

#     return render_template('dashboard.html', prediction=prediction, recommendations=recommendations)



@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if best_model is None or thresholds is None:
        flash('Model or thresholds not loaded. Please contact the administrator.')
        return redirect(url_for('dashboard'))

    prediction = None
    recommendations = []
    if request.method == 'POST':
        try:
            # Validate and convert form inputs
            data = {
                'Age': [float(request.form['age'])],
                'Gender': [request.form['gender']],
                'Tenure': [float(request.form['tenure'])],
                'Usage Frequency': [float(request.form['usage_frequency'])],
                'Support Calls': [float(request.form['support_calls'])],
                'Payment Delay': [float(request.form['payment_delay'])],
                'Subscription Type': [request.form['subscription_type']],
                'Contract Length': [request.form['contract_length']],
                'Total Spend': [float(request.form['total_spend'])],
                'Last Interaction': [float(request.form['last_interaction'])]
            }

            # Check for negative values
            numeric_fields = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
            for field in numeric_fields:
                if data[field][0] < 0:
                    raise ValueError(f"{field} cannot be negative.")

            # Validate categorical inputs
            if data['Gender'][0] not in ['Male', 'Female']:
                raise ValueError("Invalid gender value.")
            if data['Subscription Type'][0] not in ['Basic', 'Standard', 'Premium']:
                raise ValueError("Invalid subscription type.")
            if data['Contract Length'][0] not in ['Monthly', 'Quarterly', 'Annual']:
                raise ValueError("Invalid contract length.")

            customer_data = pd.DataFrame(data)

            customer_data['Recency'] = customer_data['Last Interaction'].max() - customer_data['Last Interaction']
            customer_data['Monthly Spend'] = customer_data['Total Spend'] / customer_data['Tenure'].replace(0, 1)
            customer_data['Usage Intensity'] = customer_data['Usage Frequency'] / customer_data['Tenure'].replace(0, 1)
            customer_data['Support Call Frequency'] = customer_data['Support Calls'] / customer_data['Tenure'].replace(0, 1)
            customer_data['Payment Reliability'] = 1 / (customer_data['Payment Delay'] + 1)
            contract_numeric = customer_data['Contract Length'].map(contract_mapping)
            customer_data['Tenure Contract'] = customer_data['Tenure'] * contract_numeric
            subscription_numeric = customer_data['Subscription Type'].map(subscription_mapping)
            customer_data['Subscription Spend'] = customer_data['Monthly Spend'] * subscription_numeric

            churn_prob = best_model.predict_proba(customer_data)[:, 1][0]
            prediction = round(churn_prob * 100, 2)
            recommendations = recommend_retention_strategy(customer_data, churn_prob, thresholds)

            pred = Prediction(
                user_id=current_user.id,
                prediction=churn_prob,
                date_submitted=pd.Timestamp.now(),
                input_data=json.dumps(data),
                retention_strategy=json.dumps(recommendations)
            )
            db.session.add(pred)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                flash(f'Error saving prediction: {str(e)}')
                return redirect(url_for('dashboard'))

        except ValueError as ve:
            flash(f'Invalid input: {str(ve)}')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error processing prediction: {str(e)}')
            return redirect(url_for('dashboard'))

    return render_template('dashboard.html', prediction=prediction, recommendations=recommendations)










# @app.route('/history')
# @login_required
# def history():
#     predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date_submitted.desc()).all()
#     return render_template('history.html', predictions=predictions)


@app.route('/history')
@login_required
def history():
    pred_date_start = request.args.get('pred_date_start', '')
    pred_date_end = request.args.get('pred_date_end', '')
    pred_prob_min = request.args.get('pred_prob_min', '')
    pred_prob_max = request.args.get('pred_prob_max', '')
    action = request.args.get('action', '')  # New param to trigger export


    predictions_query = Prediction.query.filter_by(user_id=current_user.id)

    # Apply date filters
    if pred_date_start:
        predictions_query = predictions_query.filter(Prediction.date_submitted >= pred_date_start)
    if pred_date_end:
        predictions_query = predictions_query.filter(Prediction.date_submitted <= pred_date_end)

    # Apply probability filters (convert percentage to float between 0-1)
    if pred_prob_min:
        predictions_query = predictions_query.filter(Prediction.prediction >= float(pred_prob_min) / 100)
    if pred_prob_max:
        predictions_query = predictions_query.filter(Prediction.prediction <= float(pred_prob_max) / 100)

    # Order by date descending
    predictions_query = predictions_query.order_by(Prediction.date_submitted.desc())


    if action == 'export':
        # Export filtered user predictions as CSV
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Date Submitted', 'Prediction (%)', 'Input Data', 'Retention Strategy'])

        for p in predictions_query.all():
            writer.writerow([
                p.date_submitted.strftime('%Y-%m-%d %H:%M:%S'),
                round(p.prediction * 100, 2),
                p.input_data,
                p.retention_strategy or ''
            ])

        output.seek(0)
        return Response(
            output,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=user_filtered_predictions.csv'}
        )

    predictions = predictions_query.all()

    return render_template('history.html',
                           predictions=predictions,
                           pred_date_start=pred_date_start,
                           pred_date_end=pred_date_end,
                           pred_prob_min=pred_prob_min,
                           pred_prob_max=pred_prob_max)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST' and 'create_username' in request.form:
        username = request.form['create_username']
        password = request.form['create_password']
        is_admin = 'create_is_admin' in request.form
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            new_user = User(username=username, password=generate_password_hash(password), is_admin=is_admin)
            db.session.add(new_user)
            try:
                db.session.commit()
                flash('User created successfully!')
            except Exception as e:
                db.session.rollback()
                flash(f'Error creating user: {str(e)}')
        return redirect(url_for('admin'))

    user_sort = request.args.get('user_sort', 'id')
    user_order = request.args.get('user_order', 'asc')
    user_filter = request.args.get('user_filter', '')
    admin_filter = request.args.get('admin_filter', 'all')
    pred_sort = request.args.get('pred_sort', 'date_submitted')
    pred_order = request.args.get('pred_order', 'desc')
    pred_user_filter = request.args.get('pred_user_filter', '')
    pred_date_start = request.args.get('pred_date_start', '')
    pred_date_end = request.args.get('pred_date_end', '')

    users_query = User.query
    if user_filter:
        users_query = users_query.filter(User.username.ilike(f'%{user_filter}%'))
    if admin_filter != 'all':
        users_query = users_query.filter(User.is_admin == (admin_filter == 'yes'))
    if user_sort == 'id':
        users = users_query.order_by(asc(User.id) if user_order == 'asc' else desc(User.id)).all()
    elif user_sort == 'username':
        users = users_query.order_by(asc(User.username) if user_order == 'asc' else desc(User.username)).all()
    elif user_sort == 'is_admin':
        users = users_query.order_by(asc(User.is_admin) if user_order == 'asc' else desc(User.is_admin)).all()
    else:
        users = users_query.order_by(asc(User.id)).all()

    preds_query = Prediction.query
    if pred_user_filter:
        preds_query = preds_query.join(User).filter(User.username.ilike(f'%{pred_user_filter}%'))
    if pred_date_start:
        preds_query = preds_query.filter(Prediction.date_submitted >= pred_date_start)
    if pred_date_end:
        preds_query = preds_query.filter(Prediction.date_submitted <= pred_date_end)
    if pred_sort == 'date_submitted':
        preds = preds_query.order_by(asc(Prediction.date_submitted) if pred_order == 'asc' else desc(Prediction.date_submitted)).all()
    elif pred_sort == 'username':
        preds = preds_query.join(User).order_by(asc(User.username) if pred_order == 'asc' else desc(User.username)).all()
    elif pred_sort == 'prediction':
        preds = preds_query.order_by(asc(Prediction.prediction) if pred_order == 'asc' else desc(Prediction.prediction)).all()
    else:
        preds = preds_query.order_by(desc(Prediction.date_submitted)).all()

    return render_template('admin.html', users=users, predictions=preds,
                          user_sort=user_sort, user_order=user_order, user_filter=user_filter, admin_filter=admin_filter,
                          pred_sort=pred_sort, pred_order=pred_order, pred_user_filter=pred_user_filter,
                          pred_date_start=pred_date_start, pred_date_end=pred_date_end)

@app.route('/view_image/<filename>')
@login_required
def view_image(filename):
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    return render_template('view_image.html', filename=filename)

@app.route('/eda', methods=['GET'])
@login_required
def eda_dashboard():
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    
    eda_images = {
        'age_distribution': 'age_distribution.png',
        'churn_distribution': 'churn_distribution.png',
        'confusion_matrix': 'confusion_matrix.png',
        'contract_churn': 'contract_churn.png',
        'correlation_matrix': 'correlation_matrix.png',
        'feature_importance': 'feature_importance.png',
        'gender_distribution': 'gender_distribution.png',
        'payment_delay_churn': 'payment_delay_churn.png',
        'subscription_churn': 'subscription_churn.png',
        'support_calls_churn': 'support_calls_churn.png',
        'tenure_churn': 'tenure_churn.png'
    }
    
    return render_template('eda_dashboard.html', eda_images=eda_images)

@app.route('/admin_user', methods=['GET', 'POST'])
@login_required
def admin_user():
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST' and 'create_username' in request.form:
        username = request.form['create_username']
        password = request.form['create_password']
        is_admin = 'create_is_admin' in request.form
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            new_user = User(username=username, password=generate_password_hash(password), is_admin=is_admin)
            db.session.add(new_user)
            try:
                db.session.commit()
                flash('User created successfully!')
            except Exception as e:
                db.session.rollback()
                flash(f'Error creating user: {str(e)}')
        return redirect(url_for('admin_user'))

    user_sort = request.args.get('user_sort', 'id')
    user_order = request.args.get('user_order', 'asc')
    user_filter = request.args.get('user_filter', '')
    admin_filter = request.args.get('admin_filter', 'all')

    users_query = User.query
    if user_filter:
        users_query = users_query.filter(User.username.ilike(f'%{user_filter}%'))
    if admin_filter != 'all':
        users_query = users_query.filter(User.is_admin == (admin_filter == 'yes'))
    
    if user_sort == 'id':
        order = User.id.asc() if user_order == 'asc' else User.id.desc()
    elif user_sort == 'username':
        order = User.username.asc() if user_order == 'asc' else User.username.desc()
    elif user_sort == 'is_admin':
        order = User.is_admin.asc() if user_order == 'asc' else User.is_admin.desc()
    else:
        order = User.id.asc()
    
    users = users_query.order_by(order).all()

    return render_template('admin_user.html',
                         users=users,
                         user_sort=user_sort,
                         user_order=user_order,
                         user_filter=user_filter,
                         admin_filter=admin_filter)
@app.route('/predictions_user')
@login_required
def predictions_user():
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))

    pred_user_filter = request.args.get('pred_user_filter', '')
    pred_date_start = request.args.get('pred_date_start', '')
    pred_date_end = request.args.get('pred_date_end', '')
    action = request.args.get('action', '')   

    preds_query = Prediction.query
    joined_user = False

    if pred_user_filter:
        preds_query = preds_query.join(User).filter(User.username.ilike(f'%{pred_user_filter}%'))
        joined_user = True

    from datetime import datetime

    if pred_date_start:
        try:
            date_start = datetime.strptime(pred_date_start, '%Y-%m-%d')
            preds_query = preds_query.filter(Prediction.date_submitted >= date_start)
        except ValueError:
            pass

    if pred_date_end:
        try:
            date_end = datetime.strptime(pred_date_end, '%Y-%m-%d')
            preds_query = preds_query.filter(Prediction.date_submitted <= date_end)
        except ValueError:
            pass

    # Always sort by latest submitted first
    preds = preds_query.order_by(Prediction.date_submitted.desc()).all()

    if action == 'export':
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['User', 'Input Data', 'Prediction', 'Date Submitted', 'Retention Strategy'])

        for p in preds:
            writer.writerow([
                p.user.username if p.user else 'Unknown',
                p.input_data,
                p.prediction,
                p.date_submitted,
                p.retention_strategy or ''
            ])

        output.seek(0)
        return Response(
            output,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=filtered_predictions.csv'}
        )

    return render_template('predictions_user.html',
                           predictions=preds,
                           pred_user_filter=pred_user_filter,
                           pred_date_start=pred_date_start,
                           pred_date_end=pred_date_end)

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        is_admin = 'is_admin' in request.form
        
        if username != user.username and User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            user.username = username
            if password:
                user.password = generate_password_hash(password)
            user.is_admin = is_admin
            try:
                db.session.commit()
                flash('User updated successfully!')
            except Exception as e:
                db.session.rollback()
                flash(f'Error updating user: {str(e)}')
        return redirect(url_for('admin_user'))
    
    return render_template('edit_user.html', user=user)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash('Admin access only!')
        return redirect(url_for('dashboard'))
    
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete yourself!')
    else:
        db.session.delete(user)
        try:
            db.session.commit()
            flash('User deleted successfully!')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting user: {str(e)}')
    return redirect(url_for('admin'))

def recommend_retention_strategy(customer_data, churn_prob, thresholds):
    recommendations = []
    importances = {
        'Support Calls': 0.210004, 'Total Spend': 0.181371, 'Age': 0.128575,
        'Support Call Frequency': 0.101341, 'Payment Reliability': 0.078405,
        'Payment Delay': 0.065760, 'Monthly Spend': 0.043016, 'Recency': 0.037757,
        'Tenure Contract': 0.037086, 'Contract Length': 0.035338, 'Tenure': 0.024439,
        'Subscription Spend': 0.018086, 'Gender_Female': 0.013695, 'Gender_Male': 0.010662,
        'Usage Intensity': 0.008391, 'Usage Frequency': 0.003268,
        'Subscription Type_Basic': 0.001950, 'Subscription Type_Standard': 0.000445,
        'Subscription Type_Premium': 0.000411
    }

    if churn_prob > 0.5:
        if customer_data['Support Calls'].values[0] > thresholds['Support Calls']:
            recommendations.append({'action': "Proactive support outreach due to high support calls.", 'priority': importances.get('Support Calls', 0)})
        if customer_data['Total Spend'].values[0] < thresholds['Total Spend']:
            recommendations.append({'action': "Offer upsell or loyalty discount to increase spend.", 'priority': importances.get('Total Spend', 0)})
        if customer_data['Payment Delay'].values[0] > thresholds['Payment Delay']:
            recommendations.append({'action': "Send payment reminder and offer flexible options.", 'priority': importances.get('Payment Delay', 0)})
        if customer_data['Payment Reliability'].values[0] < thresholds['Payment Reliability']:
            recommendations.append({'action': "Improve trust by offering payment flexibility or rewards.", 'priority': importances.get('Payment Reliability', 0)})
        if customer_data['Support Call Frequency'].values[0] > thresholds['Support Call Frequency']:
            recommendations.append({'action': "Provide self-service resources or better support resolution.", 'priority': importances.get('Support Call Frequency', 0)})
        if customer_data['Recency'].values[0] > thresholds['Recency']:
            recommendations.append({'action': "Engage user with reactivation campaigns.", 'priority': importances.get('Recency', 0)})
        if customer_data['Age'].values[0] > thresholds['Age']:
            recommendations.append({'action': "Tailor messaging for older customers, possibly include tech support.", 'priority': importances.get('Age', 0)})
        if customer_data['Monthly Spend'].values[0] < thresholds['Monthly Spend']:
            recommendations.append({'action': "Introduce usage-based offers or targeted upgrades.", 'priority': importances.get('Monthly Spend', 0)})
        if customer_data['Tenure Contract'].values[0] < thresholds['Tenure Contract']:
            recommendations.append({'action': "Encourage commitment via bundled plans or annual discounts.", 'priority': importances.get('Tenure Contract', 0)})
        contract_length_numeric = contract_mapping.get(customer_data['Contract Length'].values[0], 0)
        if contract_length_numeric < thresholds['Contract Length']:
            recommendations.append({'action': "Promote longer-term plans with better value propositions.", 'priority': importances.get('Contract Length', 0)})
        if customer_data['Tenure'].values[0] < thresholds['Tenure']:
            recommendations.append({'action': "Offer loyalty incentives to increase tenure.", 'priority': importances.get('Tenure', 0)})
        if customer_data['Subscription Spend'].values[0] < thresholds['Subscription Spend']:
            recommendations.append({'action': "Upsell to higher-tier subscriptions with relevant benefits.", 'priority': importances.get('Subscription Spend', 0)})
        if customer_data['Usage Intensity'].values[0] < thresholds['Usage Intensity']:
            recommendations.append({'action': "Educate on value/features to drive deeper engagement.", 'priority': importances.get('Usage Intensity', 0)})
        if customer_data['Usage Frequency'].values[0] < thresholds['Usage Frequency']:
            recommendations.append({'action': "Trigger engagement campaigns or gamified nudges.", 'priority': importances.get('Usage Frequency', 0)})
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
    
    return recommendations

@app.route('/admin/export')
@login_required
def admin_export_all():
    if not current_user.is_admin:
        abort(403)

    predictions = Prediction.query.all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['User', 'Input Data', 'Prediction', 'Date Submitted', 'Retention Strategy'])

    for p in predictions:
        writer.writerow([
            p.user_id,
            p.input_data,
            p.prediction,
            p.date_submitted,
            p.retention_strategy
        ])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=all_predictions.csv'}
    )



if __name__ == '__main__':
    app.run(debug=True)