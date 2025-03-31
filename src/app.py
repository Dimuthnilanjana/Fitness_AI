from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from logging.handlers import RotatingFileHandler
import smtplib
from email.mime.text import MIMEText
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.config['JSON_AS_ASCII'] = False
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('SMTP_EMAIL')
app.config['MAIL_PASSWORD'] = os.getenv('SMTP_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('SMTP_EMAIL')
mail = Mail(app)

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('workout_app')
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'workout_app.log'), maxBytes=1024*1024, backupCount=10
)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize Firebase
cred = credentials.Certificate('hdproject-6e51c-firebase-adminsdk-4e5te-d7102a3fe3.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load and preprocess datasets
bmi_df = pd.read_csv('data/bmi.csv').dropna()
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                            labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

nutrition_df = pd.read_csv('data/nutrition.csv')
columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']
for col in columns_to_normalize:
    if col in nutrition_df.columns:
        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')
    else:
        logger.warning(f"Column {col} not found in nutrition_df")
scaler = StandardScaler()
nutrition_df[columns_to_normalize] = scaler.fit_transform(nutrition_df[columns_to_normalize].fillna(0))

# Load ML model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
logger.info(f"Scaler expected feature names: {scaler.feature_names_in_}")

# Define feature names based on scaler's training data
FEATURE_NAMES = scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else [
    'weight', 'height', 'age', 'bmi', 'days_per_week', 'sleep_hours', 'calories', 
    'protein', 'carbohydrate', 'total_fat', 'fiber', 'intensity', 'exercise_type', 'rating'
]

@app.route('/')
def home():
    logger.info("Rendering home page")
    return render_template('index.html')

def generate_workout_plan(user_info):
    try:
        workouts_df = pd.read_csv('data/workouts.csv')
        equipment_map = {
            'none': 'Body Only', 'bands': 'Bands', 'barbell': 'Barbell', 'dumbbell': 'Dumbbell',
            'cable': 'Cable', 'machine': 'Machine', 'kettlebell': 'Kettlebells',
            'medicine ball': 'Medicine Ball', 'exercise ball': 'Exercise Ball'
        }
        level_map = {'1': 'Beginner', '2': 'Intermediate', '3': 'Expert'}

        equipment = equipment_map.get(user_info['equipment'].lower(), 'Body Only')
        level = level_map.get(str(user_info['fitness_level']), 'Intermediate')
        days_per_week = int(user_info.get('days_per_week', 5))
        fitness_level = int(user_info.get('fitness_level', 2))

        filtered_workouts = workouts_df[
            (workouts_df['Equipment'].str.contains(equipment, na=False)) &
            (workouts_df['Level'] == level)
        ]
        workout_groups = {
            'Cardio': filtered_workouts[filtered_workouts['Type'] == 'Cardio'].sort_values('Rating', ascending=False),
            'Strength': filtered_workouts[filtered_workouts['Type'] == 'Strength'].sort_values('Rating', ascending=False),
            'Flexibility': filtered_workouts[filtered_workouts['Type'].isin(['Stretching', 'Plyometrics'])].sort_values('Rating', ascending=False)
        }

        total_days = 30
        workout_days_per_week = min(days_per_week, 7)
        total_workout_days = (total_days / 7) * workout_days_per_week
        total_rest_days = total_days - total_workout_days
        total_rest_days = min(total_rest_days + 2, total_days - 5) if fitness_level == 1 else max(total_rest_days - 2, 2) if fitness_level == 3 else total_rest_days

        plan = {}
        rest_day_indices = set(np.random.choice(range(1, 31), size=int(total_rest_days), replace=False))
        for day in range(1, 31):
            if day in rest_day_indices:
                plan[str(day)] = {'type': 'Rest', 'exercises': [], 'intensity': 'low', 'notes': 'Focus on recovery'}
            else:
                workout_type = np.random.choice(['Cardio', 'Strength', 'Flexibility'], p=[0.4, 0.4, 0.2])
                available_exercises = workout_groups[workout_type]
                exercises = (
                    [{
                        'name': str(ex['Title']), 'desc': str(ex['Desc']), 'equipment': str(ex['Equipment']),
                        'sets': 3, 'reps': 12 if workout_type == 'Strength' else 30, 'rating': float(ex.get('Rating', 0)),
                        'intensity': user_info.get('intensity', 'moderate')
                    } for ex in available_exercises.head(3).to_dict('records')] if not available_exercises.empty else
                    [{'name': f'Basic {workout_type}', 'desc': 'Bodyweight exercise', 'equipment': 'Body Only',
                      'sets': 3, 'reps': 12, 'rating': 0, 'intensity': user_info.get('intensity', 'moderate')}]
                )
                plan[str(day)] = {'type': workout_type, 'exercises': exercises, 'intensity': user_info.get('intensity', 'moderate'), 'notes': 'Focus on form'}
        logger.info("Workout plan generated successfully")
        return plan
    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        raise

def process_form_data(form_data):
    try:
        # Define required fields based on the form
        required_fields = [
            'weight_in_kg', 'height_in_cm', 'age', 'days_per_week', 'sleep_hours', 'intensity',
            'exercise_type', 'calorie_target', 'macro_preference', 'diet_type', 'equipment',
            'fitness_level', 'meals_per_day'
        ]
        missing_fields = [field for field in required_fields if not form_data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing fields: {', '.join(missing_fields)}")

        # Convert height and weight to metric units
        height_m = float(form_data['height_in_cm']) / 100
        weight_kg = float(form_data['weight_in_kg'])
        bmi = weight_kg / (height_m ** 2)

        # Define macro ratios (consistent with generate_nutrition_plan)
        macro_ratios = {
            'balanced': {'protein': 0.3, 'carbs': 0.4, 'total_fat': 0.3},
            'high_protein': {'protein': 0.4, 'carbs': 0.4, 'total_fat': 0.2},
            'low_carb': {'protein': 0.5, 'carbs': 0.1, 'total_fat': 0.4},
            'high_carb': {'protein': 0.3, 'carbs': 0.5, 'total_fat': 0.2}
        }
        macro_pref = macro_ratios[form_data['macro_preference']]

        # Create processed_data with all required fields
        processed_data = {
            'weight': weight_kg,
            'height': height_m,
            'age': int(form_data['age']),
            'bmi': bmi,
            'days_per_week': int(form_data['days_per_week']),
            'sleep_hours': float(form_data['sleep_hours']),
            'intensity': int(form_data['intensity']),
            'exercise_type': int(form_data['exercise_type']),
            'calories': float(form_data['calorie_target']),  # Matches form field name
            'protein': macro_pref['protein'],
            'carbohydrate': macro_pref['carbs'],
            'total_fat': macro_pref['total_fat'],
            'fiber': float(form_data['calorie_target']) * (0.016 if form_data['diet_type'] == 'high_carb' else 0.014),
            'equipment': form_data['equipment'],
            'fitness_level': form_data['fitness_level'],
            'rating': 0,
            'diet_type': form_data['diet_type'],           # Added
            'macro_preference': form_data['macro_preference'],  # Added
            'meals_per_day': int(form_data['meals_per_day'])    # Added
        }
        logger.info(f"Processed form data: {processed_data}")
        return processed_data
    except Exception as e:
        logger.error(f"Error processing form data: {str(e)}")
        raise

def generate_nutrition_plan(user_data):
    try:
        logger.info(f"Generating nutrition plan with user_data: {user_data}")
        diet_type = user_data['diet_type']
        meals_per_day = user_data['meals_per_day']  # Already an int from process_form_data
        calorie_target = float(user_data['calories'])  # Use 'calories' to match processed_data
        macro_pref = user_data['macro_preference']

        # Define macro ratios (consistent with process_form_data)
        macro_ratios = {
            'balanced': {'protein': 0.3, 'carbs': 0.4, 'total_fat': 0.3},
            'high_protein': {'protein': 0.4, 'carbs': 0.4, 'total_fat': 0.2},
            'low_carb': {'protein': 0.5, 'carbs': 0.1, 'total_fat': 0.4},
            'high_carb': {'protein': 0.3, 'carbs': 0.5, 'total_fat': 0.2}
        }[macro_pref]

        # Calculate macronutrient targets
        protein_target = calorie_target * macro_ratios['protein'] / 4  # 4 kcal/g for protein
        carb_target = calorie_target * macro_ratios['carbs'] / 4       # 4 kcal/g for carbs
        fat_target = calorie_target * macro_ratios['total_fat'] / 9    # 9 kcal/g for fat
        fiber_target = calorie_target * (0.016 if diet_type == 'high_carb' else 0.014)

        # Distribute across meals
        meal_names = ['Breakfast', 'Lunch', 'Dinner'][:meals_per_day] if meals_per_day <= 3 else \
                     ['Breakfast', 'Snack 1', 'Lunch', 'Snack 2', 'Dinner'][:meals_per_day]
        meals = {
            name: {
                'calories': calorie_target / meals_per_day,
                'protein': protein_target / meals_per_day,
                'carbs': carb_target / meals_per_day,
                'fat': fat_target / meals_per_day,
                'fiber': fiber_target / meals_per_day
            } for name in meal_names
        }

        # Compile nutrition plan
        nutrition_plan = {
            'daily_targets': {
                'calories': calorie_target,
                'protein': protein_target,
                'carbs': carb_target,
                'fat': fat_target,
                'fiber': fiber_target
            },
            'meals': meals,
            'diet_type': diet_type,
            'macro_split': macro_ratios
        }
        logger.info("Nutrition plan generated successfully")
        return nutrition_plan
    except Exception as e:
        logger.error(f"Error generating nutrition plan: {str(e)}")
        raise

def format_complete_plan(workout_plan, nutrition_plan):
    """Format and combine workout and nutrition plans"""
    try:
        complete_plan = {
            'workout_plan': workout_plan,
            'nutrition_plan': nutrition_plan,
            'overview': {
                'total_days': 30,
                'workout_days': len([d for d in workout_plan if workout_plan[d]['type'] != 'Rest']),
                'rest_days': len([d for d in workout_plan if workout_plan[d]['type'] == 'Rest'])
            }
        }
        logger.info("Formatted complete plan successfully")
        return complete_plan
    except Exception as e:
        logger.error(f"Error formatting complete plan: {str(e)}")
        raise

# Admin routes remain unchanged for brevity; include them as in your current code
@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin_doc = db.collection('admins').document(username).get()
        if admin_doc.exists:
            logger.warning(f"Admin registration failed: Username {username} already exists")
            return make_response("Username already exists", 400)
        db.collection('admins').document(username).set({
            'username': username, 'password': generate_password_hash(password), 'created_at': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Admin {username} registered successfully")
        return redirect(url_for('admin_login'))
    logger.info("Rendering admin registration page")
    return render_template('admin/admin_registration.html')
# Admin login route
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin_doc = db.collection('admins').document(username).get()
        if admin_doc.exists and check_password_hash(admin_doc.to_dict()['password'], password):
            session['is_admin'] = True
            session['admin_username'] = username
            logger.info(f"Admin {username} logged in")
            return redirect(url_for('admin_dashboard'))
        logger.warning(f"Login failed for admin {username}")
        return make_response("Login failed", 401)
    logger.info("Rendering admin login page")
    return render_template('admin/admin_login.html')
# Admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'is_admin' not in session:
        logger.info("Redirecting unauthenticated user to admin login")
        return redirect(url_for('admin_login'))
    coaches = [doc.to_dict() for doc in db.collection('users').where('user_type', '==', 'coach').stream()]
    logger.info("Rendering admin dashboard")
    return render_template('admin/admin_dashboard.html', coaches=coaches)
# Admin register coach route
@app.route('/admin/register_coach', methods=['POST'])
def register_coach():
    if 'is_admin' not in session:
        logger.warning("Unauthorized attempt to register coach")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        email = request.form['email']
        password = request.form['password']
        coach_name = request.form['coach_name']
        specialization = request.form['specialization']
        profile_pic_url = request.form['profile_pic_url']
        services = request.form.getlist('services')

        user = auth.create_user(email=email, password=password)
        db.collection('users').document(user.uid).set({
            'user_type': 'coach', 'username': coach_name, 'email': email,
            'password': generate_password_hash(password), 'specialization': specialization,
            'profile_pic_url': profile_pic_url, 'services': services
        })
        logger.info(f"Coach {email} registered by admin")
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error registering coach: {str(e)}")
        return jsonify({'error': str(e)}), 400
# Admin delete coach route
@app.route('/admin/delete_coach/<coach_email>', methods=['POST'])
def delete_coach(coach_email):
    if 'is_admin' not in session:
        logger.warning("Unauthorized attempt to delete coach")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        coach_docs = db.collection('users').where('email', '==', coach_email).limit(1).get()
        if not coach_docs:
            logger.warning(f"Coach {coach_email} not found")
            return jsonify({'error': 'Coach not found'}), 404
        coach_docs[0].reference.delete()
        logger.info(f"Coach {coach_email} deleted by admin")
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error deleting coach: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Admin get coach details route
@app.route('/admin/get_coach/<coach_email>', methods=['GET'])
def get_coach(coach_email):
    if 'is_admin' not in session:
        logger.warning("Unauthorized attempt to fetch coach details")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        coach_docs = db.collection('users').where('email', '==', coach_email).where('user_type', '==', 'coach').limit(1).get()
        if not coach_docs:
            logger.warning(f"Coach {coach_email} not found")
            return jsonify({'error': 'Coach not found'}), 404
        coach_data = coach_docs[0].to_dict()
        coach_data['uid'] = coach_docs[0].id  # Include the Firestore document ID
        logger.info(f"Fetched details for coach {coach_email}")
        return jsonify(coach_data), 200
    except Exception as e:
        logger.error(f"Error fetching coach {coach_email}: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Add this new route to edit coach details
@app.route('/admin/edit_coach/<coach_email>', methods=['POST'])
def edit_coach(coach_email):
    if 'is_admin' not in session:
        logger.warning("Unauthorized attempt to edit coach")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Find the coach by email
        coach_docs = db.collection('users').where('email', '==', coach_email).where('user_type', '==', 'coach').limit(1).get()
        if not coach_docs:
            logger.warning(f"Coach {coach_email} not found")
            return jsonify({'error': 'Coach not found'}), 404
        
        coach_ref = coach_docs[0].reference
        coach_id = coach_docs[0].id

        # Get form data
        email = request.form.get('email')
        coach_name = request.form.get('coach_name')
        specialization = request.form.get('specialization')
        profile_pic_url = request.form.get('profile_pic_url')
        services = request.form.getlist('services')

        # Validate required fields
        if not all([email, coach_name, specialization, profile_pic_url]):
            logger.warning(f"Missing required fields for coach {coach_email}")
            return jsonify({'error': 'Missing required fields'}), 400

        # Update coach in Firestore
        updated_data = {
            'email': email,
            'username': coach_name,
            'specialization': specialization,
            'profile_pic_url': profile_pic_url,
            'services': services,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        coach_ref.update(updated_data)

        # Update Firebase Auth email if it changed
        if email != coach_email:
            auth.update_user(coach_id, email=email)
            logger.info(f"Updated email in Firebase Auth for coach {coach_id} from {coach_email} to {email}")

        logger.info(f"Coach {coach_email} updated successfully")
        return jsonify({'success': True, 'message': 'Coach updated successfully'}), 200
    except Exception as e:
        logger.error(f"Error editing coach {coach_email}: {str(e)}")
        return jsonify({'error': str(e)}), 500
# Email sending function
def send_email(to_email, subject, body):
    sender_email = os.getenv('SMTP_EMAIL')
    sender_password = os.getenv('SMTP_PASSWORD')
    
    if not sender_email or not sender_password:
        raise ValueError("SMTP_EMAIL and SMTP_PASSWORD must be set in .env")

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        raise
# Admin reset coach password
@app.route('/admin/reset_coach_password/<coach_email>', methods=['POST'])
def reset_coach_password(coach_email):
    if 'is_admin' not in session:
        logger.warning("Unauthorized attempt to reset coach password")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        coach_docs = db.collection('users').where('email', '==', coach_email).where('user_type', '==', 'coach').limit(1).get()
        if not coach_docs:
            logger.warning(f"Coach {coach_email} not found")
            return jsonify({'error': 'Coach not found'}), 404
        
        coach_id = coach_docs[0].id
        reset_link = auth.generate_password_reset_link(coach_email)
        logger.info(f"Password reset link generated for {coach_email}: {reset_link}")

        email_body = (
            f"Dear Coach,\n\n"
            f"A password reset has been requested for your account. Please click the link below to reset your password:\n\n"
            f"{reset_link}\n\n"
            f"If you did not request this, please ignore this email or contact support.\n\n"
            f"Best regards,\nFitness AI Team"
        )
        send_email(coach_email, "Password Reset Request", email_body)

        logger.info(f"Password reset initiated for coach {coach_email}")
        return jsonify({'success': True, 'message': 'Password reset email sent successfully'}), 200
    except Exception as e:
        logger.error(f"Error resetting password for coach {coach_email}: {str(e)}")
        return jsonify({'error': str(e)}), 500
# User registration and login routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_name = request.form['user_name']
        user_type = request.form['user_type']
        try:
            user = auth.create_user(email=email, password=password)
            db.collection('users').document(user.uid).set({
                'user_name': user_name, 'email': email, 'user_type': user_type
            })
            logger.info(f"User {email} registered as {user_type}")
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            return make_response(f"Registration failed: {str(e)}", 400)
    logger.info("Rendering registration page")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)
            user_data = db.collection('users').document(user.uid).get().to_dict()
            session['user_id'] = user.uid
            session['user_type'] = user_data['user_type']
            logger.info(f"User {email} logged in as {user_data['user_type']}")
            return redirect(url_for('customer_dashboard' if user_data['user_type'] == 'customer' else 'coach_dashboard'))
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return make_response(f"Login failed: {str(e)}", 401)
    logger.info("Rendering login page")
    return render_template('login.html')

# Password reset route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            # Check if the email exists in Firebase Authentication
            auth.get_user_by_email(email)
            
            # Generate a password reset link using Firebase Authentication
            reset_link = auth.generate_password_reset_link(email)
            logger.info(f"Password reset link generated for {email}")

            # Send the reset email using Flask-Mail
            msg = Message(
                subject="Password Reset Request",
                recipients=[email],
                body=(
                    f"Dear User,\n\n"
                    f"You have requested to reset your password. Please click the link below to reset it:\n\n"
                    f"{reset_link}\n\n"
                    f"If you did not request this, please ignore this email or contact support.\n\n"
                    f"Best regards,\nFitness AI Team"
                )
            )
            mail.send(msg)
            logger.info(f"Password reset email sent to {email}")
            
            return render_template('forgot_password.html', message="A password reset link has been sent to your email.")
        except auth.UserNotFoundError:
            logger.warning(f"Password reset requested for non-existent email: {email}")
            return render_template('forgot_password.html', error="No account found with this email.")
        except Exception as e:
            logger.error(f"Error processing password reset for {email}: {str(e)}")
            return render_template('forgot_password.html', error="Failed to send reset email. Please try again later.")
    # Render the forgot password page for GET requests
    logger.info("Rendering forgot password page")
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('home'))
# customer and coach dashboard routes
@app.route('/customer_dashboard')
def customer_dashboard():
    if 'user_id' not in session:
        logger.info("Redirecting unauthenticated user to login")
        return redirect(url_for('login'))
    user_data = db.collection('users').document(session['user_id']).get().to_dict()
    plans = [{'id': doc.id, **doc.to_dict(), 'status': doc.to_dict().get('status', 'not_sent')}
             for doc in db.collection('plans').where('user_id', '==', session['user_id']).stream()]
    logger.info("Rendering customer dashboard")
    return render_template('customer_dashboard.html', user_name=user_data.get('user_name', 'Customer'), plans=plans)

@app.route('/coach_dashboard')
def coach_dashboard():
    if 'user_id' not in session or session['user_type'] != 'coach':
        logger.info("Redirecting unauthorized user to login")
        return redirect(url_for('login'))
    plans = []
    for doc in db.collection('plans').where('status', '==', 'requested').stream():
        plan = doc.to_dict()
        plan['id'] = doc.id
        user_doc = db.collection('users').document(plan['user_id']).get()
        plan['user_name'] = user_doc.to_dict().get('user_name', 'Unknown') if user_doc.exists else 'Unknown'
        fitness_goals = {'0': 'Weight Loss', '1': 'Muscle Gain', '2': 'Endurance', '3': 'General Fitness'}
        plan['fitness_goal'] = fitness_goals.get(str(plan['user_data'].get('exercise_type', '')), 'Not specified')
        plans.append(plan)
    logger.info("Rendering coach dashboard")
    return render_template('coach_dashboard.html', plans=plans)

@app.route('/tell_coach/<plan_id>', methods=['POST'])
def send_to_coach_review(plan_id):
    if 'user_id' not in session:
        logger.warning("Unauthorized attempt to send plan to coach")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        plan_ref = db.collection('plans').document(plan_id)
        plan_doc = plan_ref.get()
        if not plan_doc.exists:
            logger.warning(f"Plan {plan_id} not found")
            return jsonify({'error': 'Plan not found'}), 404
        plan_data = plan_doc.to_dict()
        if plan_data.get('user_id') != session['user_id']:
            logger.warning(f"Unauthorized access attempt for plan {plan_id}")
            return jsonify({'error': 'Unauthorized'}), 401
        plan_ref.update({
            'status': 'requested', 'updated_at': firestore.SERVER_TIMESTAMP, 'sent_by': session['user_id']
        })
        logger.info(f"Plan {plan_id} sent to coach")
        return jsonify({'success': True, 'message': 'Plan sent to coach successfully'}), 200
    except Exception as e:
        logger.error(f"Error sending plan to coach: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/review_plan/<plan_id>', methods=['POST'])
def review_plan(plan_id):
    if 'user_id' not in session or session['user_type'] != 'coach':
        logger.warning("Unauthorized attempt to review plan")
        return jsonify({'error': 'Unauthorized'}), 401
    coach_comment = request.form.get('coach_comment')
    action = request.form.get('action')
    if not coach_comment or not action:
        logger.warning(f"Missing fields for plan {plan_id}")
        return jsonify({'error': 'Missing required fields'}), 400
    try:
        plan_ref = db.collection('plans').document(plan_id)
        plan_doc = plan_ref.get()
        if not plan_doc.exists:
            logger.warning(f"Plan {plan_id} not found")
            return jsonify({'error': 'Plan not found'}), 404
        new_status = 'approved' if action == 'approve' else 'rejected'
        plan_ref.update({
            'coach_comment': coach_comment, 'status': new_status, 'updated_at': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Plan {plan_id} {new_status} by coach")
        return jsonify({'success': True, 'status': new_status}), 200
    except Exception as e:
        logger.error(f"Error reviewing plan {plan_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_plan/<plan_id>', methods=['POST'])
def delete_plan(plan_id):
    if 'user_id' not in session or session['user_type'] != 'customer':
        logger.warning("Unauthorized attempt to delete plan")
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        plan_ref = db.collection('plans').document(plan_id)
        plan_doc = plan_ref.get()
        if not plan_doc.exists:
            logger.warning(f"Plan {plan_id} not found")
            return jsonify({'error': 'Plan not found'}), 404
        if plan_doc.to_dict().get('user_id') != session['user_id']:
            logger.warning(f"Unauthorized deletion attempt for plan {plan_id}")
            return jsonify({'error': 'Unauthorized'}), 401
        plan_ref.delete()
        logger.info(f"Plan {plan_id} deleted")
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error deleting plan {plan_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
# Generate workout and nutrition plans
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        logger.info("Rendering generate form")
        return render_template('generate_form.html')
    if 'user_id' not in session:
        logger.info("Redirecting unauthenticated user to login from generate")
        return redirect(url_for('login'))
    try:
        processed_data = process_form_data(request.form)
        features = pd.DataFrame([[
            processed_data['weight'], processed_data['height'], processed_data['age'], processed_data['bmi'],
            processed_data['days_per_week'], processed_data['sleep_hours'], processed_data['calories'],
            processed_data['protein'], processed_data['carbohydrate'], processed_data['total_fat'],
            processed_data['fiber'], processed_data['intensity'], processed_data['exercise_type'],
            processed_data['rating']
        ]], columns=FEATURE_NAMES)
        logger.info(f"Features for prediction: {features.to_dict(orient='records')}")
        
        cluster = model.predict(scaler.transform(features))[0]
        logger.info(f"Predicted cluster: {cluster}")
        
        workout_plan = generate_workout_plan(processed_data)
        logger.info("Generated workout plan")
        nutrition_plan = generate_nutrition_plan(processed_data)
        logger.info("Generated nutrition plan")
        complete_plan = format_complete_plan(workout_plan, nutrition_plan)  # Use the formatting function
        logger.info("Formatted complete plan")

        plan_data = {
            'user_id': session['user_id'], 'created_at': firestore.SERVER_TIMESTAMP, 'status': 'new',
            'workout_plan': workout_plan, 'nutrition_plan': nutrition_plan, 'user_data': processed_data,
            'cluster': int(cluster), 'coach_comment': '', 'coach_id': None
        }
        plan_ref = db.collection('plans').add(plan_data)
        logger.info("Added plan to Firestore")
        complete_plan['plan_id'] = plan_ref[1].id

        response = jsonify(complete_plan) if request.headers.get('Accept') == 'application/json' else render_template('result.html', plan=complete_plan)
        logger.info(f"Plan generated with ID {complete_plan['plan_id']}, returning response: {type(response)}")
        return response
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        if request.headers.get('Accept') == 'application/json':
            response = jsonify({'error': str(e)})
            return make_response(response, 400)
        else:
            response = render_template('error.html', error=str(e))
        return make_response(response, 400)

@app.route('/calorie_calculator')
def calorie_calculator():
    if 'user_id' not in session:
        logger.info("Redirecting unauthenticated user to login from calorie_calculator")
        return redirect(url_for('login'))
    logger.info("Rendering calorie calculator")
    return render_template('calorie-calculator.html')

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)


