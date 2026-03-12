from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for, flash, session as flask_session
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
import cv2
import os
import pandas as pd
import threading
import json
import base64
import time
from models import db, User, DetectionSession
import config as cfg
from pipeline_ui import run_pipeline_with_ui
from datetime import datetime, timezone

app = Flask(__name__)
app.config['SECRET_KEY'] = 'accident-detection-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['CLIPS_FOLDER'] = os.path.join('static', 'clips')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///accident_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLIPS_FOLDER'], exist_ok=True)

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins='*')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Prevent caching of protected pages
@app.after_request
def add_header(response):
    """Add headers to prevent caching of sensitive pages"""
    if request.endpoint and request.endpoint not in ['static', 'home', 'about', 'features', 'contact', 'landing']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Global state
detection_state = {
    'running': False,
    'preview_running': False,
    'video_source': None,
    'stats': {'total_detections': 0, 'critical_alerts': 0, 'fps': 0},
    'current_frame': None,
    'thread': None,
    'session_id': None,
    'stop_flag': False,
    'user_id': None
}

# Public routes
@app.route('/home')
def home():
    """Landing page - same as root"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/features')
def features():
    """Features page"""
    return render_template('features.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                return jsonify({'status': 'error', 'message': 'Account is deactivated'})
            
            login_user(user)
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'role': user.role,
                'redirect': url_for('admin_dashboard') if user.is_admin() else url_for('index')
            })
        
        return jsonify({'status': 'error', 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'status': 'error', 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'status': 'error', 'message': 'Email already registered'})
        
        user = User(username=username, email=email, role='user')
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'status': 'success', 'message': 'Registration successful'})
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flask_session.clear()  # Clear all session data
    response = redirect(url_for('login'))
    # Add cache control headers to logout response
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    flash('You have been logged out.', 'success')
    return response

def frame_callback(frame):
    """Callback to update current frame and emit via WebSocket"""
    detection_state['current_frame'] = frame
    
    # Encode frame to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Emit frame via WebSocket
    socketio.emit('video_frame', {'frame': frame_base64})

def stats_callback(stats):
    """Callback to update statistics and emit via WebSocket"""
    detection_state['stats'] = stats
    socketio.emit('stats_update', stats)

def run_preview_thread(video_source):
    """Run preview without detection"""
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            socketio.emit('error', {'message': 'Cannot open video source'})
            return
        
        while cap.isOpened() and not detection_state['stop_flag']:
            success, frame = cap.read()
            if not success:
                break
            
            # Just show the frame without detection
            frame_callback(frame)
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        
    except Exception as e:
        print(f"Preview error: {e}")
        socketio.emit('error', {'message': str(e)})
    finally:
        detection_state['preview_running'] = False
        detection_state['stop_flag'] = False
        socketio.emit('preview_stopped', {'message': 'Preview stopped'})

def run_detection_thread(video_source, conf_threshold, alert_severities, enable_email, enable_sms, user_id):
    """Run detection in background thread"""
    try:
        # Create user-specific clips folder
        user_clips_folder = os.path.join(app.config['CLIPS_FOLDER'], str(user_id))
        os.makedirs(user_clips_folder, exist_ok=True)
        
        # Run the actual detection pipeline
        run_pipeline_with_ui(
            video_source=video_source,
            frame_callback=frame_callback,
            stats_callback=stats_callback,
            stop_flag_func=lambda: detection_state['stop_flag'],
            session_id=detection_state['session_id'],
            user_id=user_id,
            clips_folder=user_clips_folder
        )
        
    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)})
    finally:
        # Update session on completion
        if detection_state['session_id']:
            with app.app_context():
                session = db.session.get(DetectionSession, detection_state['session_id'])
                if session:
                    session.end_time = datetime.now(timezone.utc)
                    session.status = 'completed'
                    session.total_detections = detection_state['stats']['total_detections']
                    session.critical_alerts = detection_state['stats']['critical_alerts']
                    db.session.commit()
        
        detection_state['running'] = False
        detection_state['stop_flag'] = False
        socketio.emit('detection_stopped', {'message': 'Detection completed'})

@app.route('/')
def landing():
    """Redirect to home page"""
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def index():
    """Main dashboard page"""
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('index.html', user=current_user)

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    if not current_user.is_authenticated or not current_user.is_admin():
        return redirect(url_for('login'))
    users = User.query.all()
    sessions = DetectionSession.query.order_by(DetectionSession.start_time.desc()).limit(50).all()
    return render_template('admin.html', users=users, sessions=sessions)

@socketio.on('start_preview')
def handle_start_preview(data):
    """Start video preview without detection"""
    if not current_user.is_authenticated:
        emit('error', {'message': 'Authentication required'})
        return
    
    if detection_state['running']:
        emit('error', {'message': 'Detection already running'})
        return
    
    video_source = data.get('video_source', 0)
    
    # Handle webcam
    if video_source == 'webcam':
        video_source = 0
    
    # Store user_id and video source
    user_id = current_user.id
    
    detection_state['preview_running'] = True
    detection_state['video_source'] = video_source
    detection_state['stop_flag'] = False
    detection_state['user_id'] = user_id
    
    thread = threading.Thread(target=run_preview_thread, args=(video_source,))
    thread.daemon = True
    thread.start()
    detection_state['thread'] = thread
    
    emit('preview_started', {'message': 'Preview started successfully'})

@socketio.on('stop_preview')
def handle_stop_preview():
    """Stop video preview"""
    detection_state['stop_flag'] = True
    detection_state['preview_running'] = False
    emit('preview_stopped', {'message': 'Preview stopped'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('start_detection')
def handle_start_detection(data):
    """Start detection process"""
    if not current_user.is_authenticated:
        emit('error', {'message': 'Authentication required'})
        return
    
    if detection_state['running']:
        emit('error', {'message': 'Detection already running'})
        return
    
    # Stop preview if running
    if detection_state['preview_running']:
        detection_state['stop_flag'] = True
        time.sleep(0.5)  # Wait for preview to stop
    
    video_source = data.get('video_source', 0)
    conf_threshold = float(data.get('conf_threshold', 0.4))
    alert_severities = data.get('alert_severities', ['Critical', 'High'])
    enable_email = data.get('enable_email', True)
    enable_sms = data.get('enable_sms', True)
    
    # Handle webcam
    if video_source == 'webcam':
        video_source = 0
    
    # Store user_id before thread starts
    user_id = current_user.id
    
    # Create detection session
    session = DetectionSession(
        user_id=user_id,
        video_source=str(video_source),
        status='active'
    )
    db.session.add(session)
    db.session.commit()
    
    detection_state['running'] = True
    detection_state['preview_running'] = False
    detection_state['video_source'] = video_source
    detection_state['session_id'] = session.id
    detection_state['stop_flag'] = False
    detection_state['stats'] = {'total_detections': 0, 'critical_alerts': 0, 'fps': 0}
    detection_state['user_id'] = user_id
    
    thread = threading.Thread(target=run_detection_thread, 
                             args=(video_source, conf_threshold, alert_severities, enable_email, enable_sms, user_id))
    thread.daemon = True
    thread.start()
    detection_state['thread'] = thread
    
    emit('detection_started', {'message': 'Detection started successfully'})

@socketio.on('stop_detection')
def handle_stop_detection():
    """Stop detection process"""
    detection_state['stop_flag'] = True
    detection_state['running'] = False
    
    # Update session
    if detection_state['session_id']:
        session = db.session.get(DetectionSession, detection_state['session_id'])
        if session:
            session.end_time = datetime.now(timezone.utc)
            session.status = 'stopped'
            session.total_detections = detection_state['stats']['total_detections']
            session.critical_alerts = detection_state['stats']['critical_alerts']
            db.session.commit()
    
    emit('detection_stopped', {'message': 'Detection stopped'})

@app.route('/get_logs')
@login_required
def get_logs():
    """Get accident logs for current user only"""
    if not os.path.exists(cfg.log_path):
        return jsonify([])
    
    df = pd.read_csv(cfg.log_path)
    if df.empty:
        return jsonify([])
    
    # Filter by current user's sessions
    user_sessions = DetectionSession.query.filter_by(user_id=current_user.id).all()
    session_ids = [s.id for s in user_sessions]
    
    # If session_id column exists, filter by it
    if 'session_id' in df.columns:
        df = df[df['session_id'].isin(session_ids)]
    
    return jsonify(df.to_dict('records'))

@app.route('/clear_logs', methods=['POST'])
@login_required
def clear_logs():
    """Clear accident logs (admin can clear all, users clear their own)"""
    import csv
    
    # Ensure log file exists
    if not os.path.exists(cfg.log_path):
        # Create empty log file
        os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
        with open(cfg.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'track_id', 'accident_type', 'severity', 'bbox'])
        return jsonify({'status': 'success', 'message': 'No logs to clear'})
    
    if current_user.is_admin():
        # Admin clears all logs
        with open(cfg.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'track_id', 'accident_type', 'severity', 'bbox'])
        socketio.emit('logs_cleared', {'message': 'All logs cleared'})
        return jsonify({'status': 'success', 'message': 'All logs cleared'})
    else:
        # Users can only clear their own logs
        df = pd.read_csv(cfg.log_path)
        if df.empty:
            return jsonify({'status': 'success', 'message': 'No logs to clear'})
        
        # Get user's session IDs
        user_sessions = DetectionSession.query.filter_by(user_id=current_user.id).all()
        session_ids = [s.id for s in user_sessions]
        
        # Filter out user's logs if session_id column exists
        if 'session_id' in df.columns:
            df = df[~df['session_id'].isin(session_ids)]
        
        # Write back remaining logs
        df.to_csv(cfg.log_path, index=False)
        socketio.emit('logs_cleared', {'message': 'Your logs cleared'})
        return jsonify({'status': 'success', 'message': 'Your logs cleared'})

@app.route('/get_clips')
@login_required
def get_clips():
    """Get list of saved clips for current user only"""
    user_clips_folder = os.path.join(app.config['CLIPS_FOLDER'], str(current_user.id))
    
    if not os.path.exists(user_clips_folder):
        return jsonify([])
    
    clips = [f for f in os.listdir(user_clips_folder) if f.endswith('.mp4')]
    return jsonify(clips)

@app.route('/clips/<int:user_id>/<filename>')
@login_required
def serve_clip(user_id, filename):
    """Serve video clip"""
    # Users can only access their own clips, admins can access all
    if not current_user.is_admin() and user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    user_clips_folder = os.path.join(app.config['CLIPS_FOLDER'], str(user_id))
    return send_from_directory(user_clips_folder, filename)

@app.route('/uploads/<int:user_id>/<filename>')
@login_required
def serve_upload(user_id, filename):
    """Serve uploaded video"""
    # Users can only access their own uploads, admins can access all
    if not current_user.is_admin() and user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    return send_from_directory(user_upload_folder, filename)

@app.route('/delete_clip/<filename>', methods=['POST'])
@login_required
def delete_clip(filename):
    """Delete a clip (only user's own clips)"""
    # Verify clip belongs to current user
    user_sessions = DetectionSession.query.filter_by(user_id=current_user.id).all()
    session_ids = [str(s.id) for s in user_sessions]
    
    # Check if clip belongs to user
    belongs_to_user = False
    for sid in session_ids:
        if filename.startswith(sid + '_'):
            belongs_to_user = True
            break
    
    if not belongs_to_user:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    clip_path = os.path.join(cfg.clip_output_dir, filename)
    if os.path.exists(clip_path):
        os.remove(clip_path)
        socketio.emit('clip_deleted', {'filename': filename})
        return jsonify({'status': 'success', 'message': f'Deleted {filename}'})
    return jsonify({'status': 'error', 'message': 'Clip not found'})

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    """Upload video file"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Create user-specific upload folder
    user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    os.makedirs(user_upload_folder, exist_ok=True)
    
    filename = file.filename
    filepath = os.path.join(user_upload_folder, filename)
    file.save(filepath)
    
    return jsonify({'status': 'success', 'filepath': filepath})

@app.route('/get_statistics')
@login_required
def get_statistics():
    """Get accident statistics for current user only"""
    if not os.path.exists(cfg.log_path):
        return jsonify({'accident_types': {}, 'severities': {}})
    
    df = pd.read_csv(cfg.log_path)
    if df.empty:
        return jsonify({'accident_types': {}, 'severities': {}})
    
    # Filter by current user's sessions
    user_sessions = DetectionSession.query.filter_by(user_id=current_user.id).all()
    session_ids = [s.id for s in user_sessions]
    
    if 'session_id' in df.columns:
        df = df[df['session_id'].isin(session_ids)]
    
    if df.empty:
        return jsonify({'accident_types': {}, 'severities': {}})
    
    accident_counts = df['accident_type'].value_counts().to_dict()
    severity_counts = df['severity'].value_counts().to_dict()
    
    return jsonify({
        'accident_types': accident_counts,
        'severities': severity_counts
    })

# Admin routes
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'role': u.role,
        'is_active': u.is_active,
        'created_at': u.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'last_login': u.last_login.strftime('%Y-%m-%d %H:%M:%S') if u.last_login else 'Never'
    } for u in users])

@app.route('/admin/user/<int:user_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_user(user_id):
    """Toggle user active status"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    if user.is_admin():
        return jsonify({'status': 'error', 'message': 'Cannot deactivate admin users'})
    
    user.is_active = not user.is_active
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': f'User {"activated" if user.is_active else "deactivated"}',
        'is_active': user.is_active
    })

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete user"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    if user.is_admin():
        return jsonify({'status': 'error', 'message': 'Cannot delete admin users'})
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'User deleted'})

@app.route('/admin/activity')
@login_required
@admin_required
def admin_activity():
    """Get recent activity feed"""
    activities = []
    
    # Get recent sessions (last 10)
    recent_sessions = DetectionSession.query.order_by(DetectionSession.start_time.desc()).limit(10).all()
    for session in recent_sessions:
        if session.critical_alerts > 0:
            activities.append({
                'type': 'critical',
                'icon': '🚨',
                'message': f'Critical Alert: {session.critical_alerts} critical incidents detected',
                'user': session.user.username,
                'time': session.start_time
            })
        else:
            activities.append({
                'type': 'info',
                'icon': '📹',
                'message': f'Session Started: {session.user.username} started detection',
                'user': session.user.username,
                'time': session.start_time
            })
    
    # Get recent users (last 5)
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    for user in recent_users:
        activities.append({
            'type': 'success',
            'icon': '✅',
            'message': f'New User: {user.username} registered',
            'user': user.username,
            'time': user.created_at
        })
    
    # Sort by time (most recent first)
    activities.sort(key=lambda x: x['time'], reverse=True)
    
    # Format time as relative (e.g., "2 minutes ago")
    now = datetime.now(timezone.utc)
    
    for activity in activities[:15]:  # Limit to 15 most recent
        time_diff = now - activity['time'].replace(tzinfo=timezone.utc) if activity['time'].tzinfo is None else now - activity['time']
        
        seconds = time_diff.total_seconds()
        if seconds < 60:
            activity['time_ago'] = 'Just now'
        elif seconds < 3600:
            minutes = int(seconds / 60)
            activity['time_ago'] = f'{minutes} minute{"s" if minutes != 1 else ""} ago'
        elif seconds < 86400:
            hours = int(seconds / 3600)
            activity['time_ago'] = f'{hours} hour{"s" if hours != 1 else ""} ago'
        else:
            days = int(seconds / 86400)
            activity['time_ago'] = f'{days} day{"s" if days != 1 else ""} ago'
        
        # Remove the datetime object (not JSON serializable)
        del activity['time']
    
    return jsonify(activities[:15])

@app.route('/admin/sessions')
@login_required
@admin_required
def admin_sessions():
    """Get all detection sessions"""
    try:
        sessions = DetectionSession.query.order_by(DetectionSession.start_time.desc()).all()
        return jsonify([{
            'id': s.id,
            'username': s.user.username,
            'video_source': s.video_source,
            'start_time': s.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': s.end_time.strftime('%Y-%m-%d %H:%M:%S') if s.end_time else 'Running',
            'total_detections': s.total_detections,
            'critical_alerts': s.critical_alerts,
            'status': s.status
        } for s in sessions])
    except Exception as e:
        print(f"Error in admin_sessions: {e}")
        return jsonify([]), 200

@app.route('/admin/logs')
@login_required
@admin_required
def admin_logs():
    """Get all detection logs"""
    if not os.path.exists(cfg.log_path):
        return jsonify([])
    
    df = pd.read_csv(cfg.log_path)
    return jsonify(df.to_dict('records'))

@app.route('/admin/clips')
@login_required
@admin_required
def admin_clips():
    """Get all video clips with user info"""
    clips = []
    clips_base = app.config['CLIPS_FOLDER']
    
    if not os.path.exists(clips_base):
        return jsonify([])
    
    # Iterate through user folders
    for user_folder in os.listdir(clips_base):
        user_folder_path = os.path.join(clips_base, user_folder)
        if os.path.isdir(user_folder_path):
            user_id = int(user_folder)
            user = db.session.get(User, user_id)
            username = user.username if user else 'Unknown'
            
            for filename in os.listdir(user_folder_path):
                if filename.endswith('.mp4'):
                    clips.append({
                        'filename': filename,
                        'user_id': user_id,
                        'username': username
                    })
    
    return jsonify(clips)

@app.route('/admin/clip/<int:user_id>/<filename>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_clip(user_id, filename):
    """Delete any clip (admin only)"""
    user_clips_folder = os.path.join(app.config['CLIPS_FOLDER'], str(user_id))
    clip_path = os.path.join(user_clips_folder, filename)
    
    if os.path.exists(clip_path):
        os.remove(clip_path)
        return jsonify({'status': 'success', 'message': f'Deleted {filename}'})
    return jsonify({'status': 'error', 'message': 'Clip not found'})

@app.route('/admin/clips/delete-all', methods=['POST'])
@login_required
@admin_required
def admin_delete_all_clips():
    """Delete all clips (admin only)"""
    clips_base = app.config['CLIPS_FOLDER']
    
    if not os.path.exists(clips_base):
        return jsonify({'status': 'success', 'message': 'No clips to delete'})
    
    deleted_count = 0
    for user_folder in os.listdir(clips_base):
        user_folder_path = os.path.join(clips_base, user_folder)
        if os.path.isdir(user_folder_path):
            for filename in os.listdir(user_folder_path):
                if filename.endswith('.mp4'):
                    clip_path = os.path.join(user_folder_path, filename)
                    try:
                        os.remove(clip_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
    
    return jsonify({'status': 'success', 'message': f'Deleted {deleted_count} clip(s)'})

@app.route('/admin/uploads')
@login_required
@admin_required
def admin_uploads():
    """Get all uploaded videos with user info"""
    uploads = []
    uploads_base = app.config['UPLOAD_FOLDER']
    
    if not os.path.exists(uploads_base):
        return jsonify([])
    
    # Iterate through user folders
    for user_folder in os.listdir(uploads_base):
        user_folder_path = os.path.join(uploads_base, user_folder)
        if os.path.isdir(user_folder_path):
            user_id = int(user_folder)
            user = db.session.get(User, user_id)
            username = user.username if user else 'Unknown'
            
            for filename in os.listdir(user_folder_path):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    filepath = os.path.join(user_folder_path, filename)
                    file_stat = os.stat(filepath)
                    
                    uploads.append({
                        'filename': filename,
                        'user_id': user_id,
                        'user': username,
                        'size': file_stat.st_size,
                        'uploaded': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    })
    
    return jsonify(uploads)

@app.route('/admin/upload/<int:user_id>/<filename>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_upload(user_id, filename):
    """Delete uploaded video (admin only)"""
    user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    upload_path = os.path.join(user_upload_folder, filename)
    
    if os.path.exists(upload_path):
        os.remove(upload_path)
        return jsonify({'status': 'success', 'message': f'Deleted {filename}'})
    return jsonify({'status': 'error', 'message': 'File not found'})

@app.route('/admin/uploads/delete-all', methods=['POST'])
@login_required
@admin_required
def admin_delete_all_uploads():
    """Delete all uploaded videos (admin only)"""
    uploads_base = app.config['UPLOAD_FOLDER']
    
    if not os.path.exists(uploads_base):
        return jsonify({'status': 'success', 'message': 'No uploads to delete'})
    
    deleted_count = 0
    for user_folder in os.listdir(uploads_base):
        user_folder_path = os.path.join(uploads_base, user_folder)
        if os.path.isdir(user_folder_path):
            for filename in os.listdir(user_folder_path):
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    upload_path = os.path.join(user_folder_path, filename)
                    try:
                        os.remove(upload_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
    
    return jsonify({'status': 'success', 'message': f'Deleted {deleted_count} upload(s)'})

@app.route('/user/uploads')
@login_required
def user_uploads():
    """Get current user's uploaded videos"""
    user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    
    if not os.path.exists(user_upload_folder):
        return jsonify([])
    
    uploads = []
    for filename in os.listdir(user_upload_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            filepath = os.path.join(user_upload_folder, filename)
            file_stat = os.stat(filepath)
            uploads.append({
                'filename': filename,
                'size': file_stat.st_size,
                'uploaded': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return jsonify(uploads)

@app.route('/user/stats')
@login_required
def user_stats():
    """Get current user's statistics"""
    sessions = DetectionSession.query.filter_by(user_id=current_user.id).all()
    
    total_detections = sum(s.total_detections or 0 for s in sessions)
    critical_alerts = sum(s.critical_alerts or 0 for s in sessions)
    total_sessions = len(sessions)
    
    return jsonify({
        'total_detections': total_detections,
        'critical_alerts': critical_alerts,
        'total_sessions': total_sessions
    })

@app.route('/user/sessions')
@login_required
def user_sessions():
    """Get current user's detection sessions"""
    sessions = DetectionSession.query.filter_by(user_id=current_user.id).order_by(DetectionSession.start_time.desc()).all()
    return jsonify([{
        'id': s.id,
        'video_source': s.video_source,
        'start_time': s.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': s.end_time.strftime('%Y-%m-%d %H:%M:%S') if s.end_time else 'Running',
        'total_detections': s.total_detections or 0,
        'critical_alerts': s.critical_alerts or 0,
        'status': s.status
    } for s in sessions])

@app.route('/user/session/<int:session_id>/stop', methods=['POST'])
@login_required
def stop_user_session(session_id):
    """Stop a specific session"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    # Users can only stop their own sessions, admins can stop any
    if not current_user.is_admin() and session.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    if session.status == 'active':
        session.end_time = datetime.now(timezone.utc)
        session.status = 'stopped'
        db.session.commit()
        
        # If this is the currently running session, stop it
        if detection_state['session_id'] == session_id:
            detection_state['stop_flag'] = True
            detection_state['running'] = False
    
    return jsonify({'status': 'success', 'message': 'Session stopped'})

@app.route('/user/session/<int:session_id>/delete', methods=['POST'])
@login_required
def delete_user_session(session_id):
    """Delete a specific session (user can only delete their own)"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    # Users can only delete their own sessions
    if session.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    # Cannot delete active sessions
    if session.status == 'active':
        return jsonify({'status': 'error', 'message': 'Cannot delete active session. Stop it first.'}), 400
    
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Session deleted'})

@app.route('/user/session/<int:session_id>', methods=['GET'])
@login_required
def view_user_session(session_id):
    """View session details"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    # Users can only view their own sessions
    if session.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    return jsonify({
        'id': session.id,
        'video_source': session.video_source,
        'start_time': session.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'Running',
        'total_detections': session.total_detections or 0,
        'critical_alerts': session.critical_alerts or 0,
        'status': session.status
    })

@app.route('/admin/session/<int:session_id>/stop', methods=['POST'])
@login_required
@admin_required
def admin_stop_session(session_id):
    """Admin stops any session"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    if session.status == 'active':
        session.end_time = datetime.now(timezone.utc)
        session.status = 'stopped'
        db.session.commit()
        
        # If this is the currently running session, stop it
        if detection_state['session_id'] == session_id:
            detection_state['stop_flag'] = True
            detection_state['running'] = False
    
    return jsonify({'status': 'success', 'message': f'Session {session_id} stopped'})

@app.route('/admin/session/<int:session_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_session(session_id):
    """Admin deletes any session"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    # Cannot delete active sessions
    if session.status == 'active':
        return jsonify({'status': 'error', 'message': 'Cannot delete active session. Stop it first.'}), 400
    
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': f'Session {session_id} deleted'})

@app.route('/admin/session/<int:session_id>', methods=['GET'])
@login_required
@admin_required
def view_admin_session(session_id):
    """Admin views any session details"""
    session = db.session.get(DetectionSession, session_id)
    
    if not session:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    return jsonify({
        'id': session.id,
        'username': session.user.username,
        'user_id': session.user_id,
        'video_source': session.video_source,
        'start_time': session.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'Running',
        'total_detections': session.total_detections or 0,
        'critical_alerts': session.critical_alerts or 0,
        'status': session.status
    })

@app.route('/admin/sessions/stop-all', methods=['POST'])
@login_required
@admin_required
def admin_stop_all_sessions():
    """Admin stops all active sessions"""
    active_sessions = DetectionSession.query.filter_by(status='active').all()
    
    if not active_sessions:
        return jsonify({'status': 'success', 'message': 'No active sessions to stop'})
    
    stopped_count = len(active_sessions)
    for session in active_sessions:
        session.end_time = datetime.now(timezone.utc)
        session.status = 'stopped'
    
    if detection_state['running']:
        detection_state['stop_flag'] = True
        detection_state['running'] = False
    
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': f'Stopped {stopped_count} active session(s)'})

@app.route('/admin/sessions/delete-all', methods=['POST'])
@login_required
@admin_required
def admin_delete_all_sessions():
    """Admin stops and deletes all sessions"""
    # Stop all active sessions first
    active_sessions = DetectionSession.query.filter_by(status='active').all()
    for session in active_sessions:
        session.end_time = datetime.now(timezone.utc)
        session.status = 'stopped'
    
    if detection_state['running']:
        detection_state['stop_flag'] = True
        detection_state['running'] = False
    
    db.session.commit()
    
    # Delete all sessions
    all_sessions = DetectionSession.query.all()
    count = len(all_sessions)
    
    for session in all_sessions:
        db.session.delete(session)
    
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': f'Stopped active sessions and deleted all {count} session(s)'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create default admin if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', email='admin@accident-detection.com', role='admin')
            admin.set_password('admin123')  # Change this in production!
            db.session.add(admin)
            db.session.commit()
            print('✓ Default admin created: username=admin, password=admin123')
        
        # Create demo user if not exists
        demo_user = User.query.filter_by(username='demo').first()
        if not demo_user:
            demo_user = User(username='demo', email='demo@accident-detection.com', role='user')
            demo_user.set_password('demo123')  # Change this in production!
            db.session.add(demo_user)
            db.session.commit()
            print('✓ Demo user created: username=demo, password=demo123')
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
