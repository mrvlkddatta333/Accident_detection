# ---------------------------
# Email Alert with Optional Video Attachment
# ---------------------------
from config import *

def send_email_alert(subject, body, to_email, video_path=None):
<<<<<<< HEAD
=======
    if not ALERTS_ENABLED:
        print("Email alert skipped: credentials not configured")
        return
>>>>>>> master
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

<<<<<<< HEAD
    # Attach video if exists
    if video_path and os.path.exists(video_path):
=======
    # Attach video if exists - with path traversal protection
    if video_path and os.path.exists(video_path):
        # Sanitize path to prevent path traversal attacks
        video_path = os.path.abspath(video_path)
        if not video_path.startswith(os.path.abspath(clip_output_dir)):
            print(f"Security: Rejected video path outside clips directory: {video_path}")
            return
            
>>>>>>> master
        with open(video_path, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(video_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(video_path)}"'
            msg.attach(part)

    try:
<<<<<<< HEAD
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
=======
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, app_password)
            server.send_message(msg)
>>>>>>> master
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)


# ---------------------------
# SMS Alert via Twilio
# ---------------------------
def send_sms_alert(track_id, frame_idx, accident_type, severity, class_name):
<<<<<<< HEAD
=======
    if not ALERTS_ENABLED:
        print("SMS alert skipped: credentials not configured")
        return
        
>>>>>>> master
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clip_name = f"ID{track_id}_{accident_type}.mp4"

    sms_message = (
        f"🚨 {accident_type} ({severity})\n"
        f"Object: {class_name} (ID {track_id})\n"
        f"Frame: {frame_idx}\n"
        f"Clip: {clip_name}\n"
<<<<<<< HEAD
        f"Time: {timestamp}"
=======
        f"Time: {timestamp}\n"
>>>>>>> master
        "Check your Mail for Video Clip"
    )

    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=sms_message,
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"SMS sent! SID: {message.sid}")
    except Exception as e:
        print("Failed to send SMS:", e)
