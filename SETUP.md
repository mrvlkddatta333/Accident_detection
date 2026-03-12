# Setup Instructions

## Environment Configuration

### 1. Create .env File

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

### 2. Configure Credentials

Edit the `.env` file and add your actual credentials:

```env
# Twilio Configuration
TWILIO_SID=your_actual_twilio_sid
TWILIO_AUTH_TOKEN=your_actual_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
TO_PHONE_NUMBER=+1234567890

# Email Configuration
FROM_EMAIL=your_email@gmail.com
EMAIL_APP_PASSWORD=your_gmail_app_password
ALERT_RECIPIENT_EMAIL=recipient@example.com
```

### 3. Gmail App Password

To get a Gmail App Password:
1. Go to your Google Account settings
2. Enable 2-Factor Authentication
3. Go to Security > 2-Step Verification > App passwords
4. Generate a new app password for "Mail"
5. Copy the 16-character password to your .env file

Reference: https://support.google.com/accounts/answer/185833

### 4. Twilio Setup

1. Sign up at https://www.twilio.com/
2. Get your Account SID and Auth Token from the console
3. Get a Twilio phone number
4. Verify your recipient phone number (required for trial accounts)

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Setup

Run the application. If credentials are missing, you'll see a warning:

```
WARNING: Missing environment variables: TWILIO_SID, FROM_EMAIL
Alert functionality will be disabled. Set these in .env file or environment.
```

## Security Notes

- **NEVER** commit the `.env` file to version control
- The `.gitignore` file is configured to exclude `.env`
- Rotate credentials regularly
- Use environment-specific credentials (dev/prod)
- Consider using AWS Secrets Manager or similar for production

## Running the Application

```bash
python main.py
```

For live detection from webcam, edit `main.py`:

```python
video_input = 0  # Use 0 for default webcam
```
