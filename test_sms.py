from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
from_number = os.getenv('TWILIO_PHONE_NUMBER')
to_number = os.getenv('TO_PHONE_NUMBER')

client = Client(account_sid, auth_token)

print(f"Sending test SMS from {from_number} to {to_number}...")

try:
    message = client.messages.create(
        from_=from_number,
        body='Test message from Accident Detection System - Setup Complete!',
        to=to_number
    )
    print(f"✓ SMS sent successfully!")
    print(f"Message SID: {message.sid}")
    print(f"Status: {message.status}")
    
    # Wait a moment and check status
    import time
    time.sleep(3)
    
    updated_message = client.messages(message.sid).fetch()
    print(f"Updated Status: {updated_message.status}")
    
    if updated_message.status == 'delivered':
        print("✓ Message delivered successfully!")
    elif updated_message.status == 'sent':
        print("✓ Message sent, waiting for delivery...")
    elif updated_message.status == 'failed':
        print(f"✗ Message failed - Error Code: {updated_message.error_code}")
        print(f"Error Message: {updated_message.error_message}")
    
except Exception as e:
    print(f"✗ Failed to send SMS: {e}")
