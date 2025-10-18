import os
import cv2
import time
import numpy as np
import joblib
import tensorflow as tf
from functools import wraps
from fpdf import FPDF
import mediapipe as mp
from flask import Flask, render_template, Response, request, jsonify, session, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId 
import os
os.environ["KERAS_BACKEND"] = "jax" 
import keras  

# --- App Configuration ---
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = "uploads"
VIDEO_SAVE_PATH = "downloads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# --- MongoDB Setup ---
app.config['MONGO_URI'] = ('mongodb+srv://sandaruiit_test:123456test@cluster0.x62vwfs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
client = MongoClient(
    app.config['MONGO_URI'],
    tls=True,
    tlsAllowInvalidCertificates=False,
    serverSelectionTimeoutMS=5000  # 5 second timeout
)
    
# Test the connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"MongoDB connection error: {str(e)}")
    
# Get database and collection
db = client.get_database("badminton")
players = db.players  
statistics = db.statistics  

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="sandarubandara/BadmintonFlex", filename="epoch_10_valacc_1.00.keras")
model = keras.saving.load_model(model_path)
try:
    model_reg = joblib.load("pose_regressor.pkl")
    scaler = joblib.load("pose_scaler.pkl")
except Exception as e:
    print("Error loading models:", e)

# --- Session State ---
selected_shot = None
correct_count = 0
total_count = 0
countdown_active = False
shot_active = False
countdown_start_time = 0
shot_start_time = 0
session_finished = False

# --- MediaPipe Init ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Shot Labels and Angles ---
SHOT_LABELS = ['backhand_drive', 'forehand_clear', 'forehand_drive', 'forehand_lift', 'forehand_net_shot']
SHOT_ANGLES = {
    "backhand_drive": {"elbow": (10, 140), "shoulder": (80, 170)},
    "forehand_lift": {"elbow": (30, 180), "shoulder": (135, 180)},
    "backhand_net_shot": {"elbow": (85, 125), "shoulder": (25, 85)},
    "forehand_net_shot": {"elbow": (120, 180), "shoulder": (150, 175)},
    "forehand_drive": {"elbow": (120, 180), "shoulder": (150, 170)},
    "forehand_clear": {"elbow": (120, 175), "shoulder": (65, 135)}
}

# --- Login Decorator ---
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash("Login required.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# --- Frame Utilities ---
SEQUENCE_LENGTH = 20
IMG_SIZE = (224, 224)

def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, SEQUENCE_LENGTH, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
    finally:
        cap.release()
    return np.array(frames)

def preprocess_sequence(frames):
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
    return np.expand_dims(frames[:SEQUENCE_LENGTH], axis=0)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return round(360 - angle if angle > 180 else angle, 2)

def generate_preview_image(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# --- User Management Functions ---
def get_user_data():
    """Get current user data from database"""
    if 'user_id' in session:
        user = players.find_one({"_id": ObjectId(session['user_id'])})
        return user
    return None

def get_user_statistics():
    """Get statistics for the current user"""
    if 'user_id' in session:
        stats = statistics.find({"user_id": ObjectId(session['user_id'])}).sort("timestamp", -1)
        return list(stats)
    return []

# In get_user_shot_statistics function:
def get_user_shot_statistics():
    """Get aggregated shot statistics for the current user"""
    if 'user_id' in session:
        pipeline = [
            {"$match": {"user_id": ObjectId(session['user_id'])}},  # Corrected line
            {"$group": {
                "_id": "$shot",
                "total_sessions": {"$sum": 1},
                "total_attempts": {"$sum": "$total"},
                "total_correct": {"$sum": "$correct"},
                "avg_accuracy": {"$avg": {"$divide": [{"$multiply": ["$correct", 100]}, "$total"]}}
            }},
            {"$sort": {"total_sessions": -1}}
        ]
        shot_stats = list(statistics.aggregate(pipeline))
        return shot_stats
    return []

# --- Prediction ---
@app.route("/predict", methods=["POST"])
@login_required
def predict_shot():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    try:
        frames = extract_sequence(video_path)
        sequence = preprocess_sequence(frames)
        pred = model.predict(sequence, verbose=0)[0]
        class_index = np.argmax(pred)
        confidence = float(round(pred[class_index] * 100, 2))
        class_name = SHOT_LABELS[class_index]
        preview_image = generate_preview_image(video_path)
        
        # Save prediction to user history
        if 'user_id' in session:
            players.update_one(
                {"_id": session['user_id']},
                {"$push": {
                    "predictions": {
                        "shot": class_name,
                        "accuracy": confidence,
                        "timestamp": datetime.now()
                    }
                }}
            )
        
        return jsonify({
            "prediction": class_name,
            "accuracy": confidence,
            "message": "Shot classification successful!",
            "preview_image": preview_image
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})
    finally:
        os.remove(video_path)

# --- PDF Summary ---
def generate_pdf_summary():
    accuracy = (correct_count / total_count * 100) if total_count else 0
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Badminton Shot Summary", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    
    # Add player name
    user = get_user_data()
    if user:
        pdf.cell(200, 10, f"Player: {user['name']}", ln=True)
    
    pdf.cell(200, 10, f"Shot: {selected_shot}", ln=True)
    pdf.cell(200, 10, f"Correct Shots: {correct_count}", ln=True)
    pdf.cell(200, 10, f"Total Attempts: {total_count}", ln=True)
    pdf.cell(200, 10, f"Accuracy: {accuracy:.2f}%", ln=True)
    pdf.output(os.path.join(VIDEO_SAVE_PATH, "summary.pdf"))

# --- Live Feed ---
def generate_frames():
    global countdown_active, shot_active, countdown_start_time, shot_start_time, total_count, correct_count
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        color = (0, 0, 255)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if countdown_active:
            elapsed = int(time.time() - countdown_start_time)
            if elapsed < 4:
                text = str(elapsed + 1) if elapsed < 3 else "Start!"
                cv2.putText(frame, text, (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
            else:
                countdown_active = False
                shot_active = True
                shot_start_time = time.time()

        elif shot_active and time.time() - shot_start_time > 2:
            shot_active = False

        elif shot_active and results.pose_landmarks and selected_shot:
            lm = results.pose_landmarks.landmark
            l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
            l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
            l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h]

            elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
            shoulder = calculate_angle(l_elbow, l_shoulder, [l_shoulder[0], l_shoulder[1] - 100])
            total_count += 1

            angles = SHOT_ANGLES[selected_shot]
            if angles['elbow'][0] <= elbow <= angles['elbow'][1] and angles['shoulder'][0] <= shoulder <= angles['shoulder'][1]:
                correct_count += 1
                color = (0, 255, 0)

            cv2.putText(frame, f"Elbow: {elbow}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Shoulder: {shoulder}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=6, circle_radius=6),
                mp_drawing.DrawingSpec(color=color, thickness=6, circle_radius=6)
            )

        cv2.putText(frame, f"Shot: {selected_shot}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Correct: {correct_count}/{total_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# ------------------ ROUTES ------------------
@app.route("/")
@login_required
def home():
    user = get_user_data()
    return render_template("home.html", user=user)

@app.route("/live_session")
@login_required
def live_session():
    user = get_user_data()
    return render_template("live_session.html", user=user)

@app.route("/about")
def about():
    user = get_user_data()
    return render_template("about.html", user=user)

@app.route("/contact")
def contact():
    user = get_user_data()
    return render_template("contact.html", user=user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user by email
        user = players.find_one({"email": email})
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials", "danger")
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')
    name = request.form.get('name', email.split('@')[0])  # Default to username from email
    
    # Check if user already exists
    existing_user = players.find_one({"email": email})
    if existing_user:
        flash("Email already registered", "danger")
        return redirect(url_for('login'))
    
    # Create new user
    hashed_password = generate_password_hash(password)
    user_id = players.insert_one({
        "email": email,
        "password": hashed_password,
        "name": name,
        "joined_date": datetime.now(),
        "predictions": [],
        "sessions": []
    }).inserted_id

    # In the signup route
    user_id = players.insert_one({...}).inserted_id
    session['user_id'] = str(user_id) 
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# In get_user_shot_statistics function:
def get_user_shot_statistics():
    """Get aggregated shot statistics for the current user"""
    if 'user_id' in session:
        pipeline = [
            {"$match": {"user_id": ObjectId(session['user_id'])}},  # Corrected line
            {"$group": {
                "_id": "$shot",
                "total_sessions": {"$sum": 1},
                "total_attempts": {"$sum": "$total"},
                "total_correct": {"$sum": "$correct"},
                "avg_accuracy": {"$avg": {"$divide": [{"$multiply": ["$correct", 100]}, "$total"]}}
            }},
            {"$sort": {"total_sessions": -1}}
        ]
        shot_stats = list(statistics.aggregate(pipeline))
        return shot_stats
    return []

# In the profile route's overall_accuracy pipeline:
@app.route('/profile')
@login_required
def profile():
    user = get_user_data()
    shot_stats = get_user_shot_statistics()
    
    # Correct recent_sessions query
    recent_sessions = list(statistics.find({"user_id": ObjectId(session['user_id'])}).sort("timestamp", -1).limit(5))
    
    # Calculate overall statistics
    total_sessions = statistics.count_documents({"user_id": ObjectId(session['user_id'])})
    
    # Correct pipeline for overall_accuracy
    pipeline = [
        {"$match": {"user_id": ObjectId(session['user_id'])}},
        {"$group": {
            "_id": None,
            "total_correct": {"$sum": "$correct"},
            "total_attempts": {"$sum": "$total"}
        }}
    ]
    overall_stats = list(statistics.aggregate(pipeline))
    
    # Rest of the code remains the same...
    if overall_stats and overall_stats[0]['total_attempts'] > 0:
        overall_accuracy = (overall_stats[0]['total_correct'] / overall_stats[0]['total_attempts']) * 100
    
    return render_template(
        'profile.html', 
        user=user, 
        shot_stats=shot_stats,
        recent_sessions=recent_sessions,
        total_sessions=total_sessions,
        overall_accuracy=overall_accuracy
    )

@app.route('/select_shot', methods=['POST'])
def select_shot():
    global selected_shot, correct_count, total_count
    selected_shot = request.json.get("shot")
    correct_count = 0
    total_count = 0
    return jsonify({"message": f"Shot selected: {selected_shot}!"})

@app.route('/start_countdown', methods=['POST'])
def start_countdown():
    global countdown_active, countdown_start_time
    countdown_active = True
    countdown_start_time = time.time()
    return jsonify({"message": "Countdown started."})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finish_session', methods=['POST'])
def finish_session():
    global session_finished, selected_shot, correct_count, total_count
    session_finished = True
    generate_pdf_summary()
    accuracy = (correct_count / total_count * 100) if total_count else 0
    
    # Save session statistics to database
    if 'user_id' in session and total_count > 0:
        statistics.insert_one({
            "user_id": ObjectId(session['user_id']),
            "shot": selected_shot,
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy,
            "timestamp": datetime.now()
        })
        
        # Update player's all-time statistics
        players.update_one(
            {"_id": ObjectId(session['user_id'])},  # Convert to ObjectId
            {"$push": {
                "sessions": {
                    "shot": selected_shot,
                    "correct": correct_count,
                    "total": total_count,
                    "accuracy": accuracy,
                    "timestamp": datetime.now()
                }
            }}
        )
    
    return jsonify({
        "message": "Session finished",
        "shot": selected_shot,
        "correct": correct_count,
        "total": total_count,
        "accuracy": f"{accuracy:.2f}",
        "pdf_url": "/downloads/summary.pdf"
    })

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(VIDEO_SAVE_PATH, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)