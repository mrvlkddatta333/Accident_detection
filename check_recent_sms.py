from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

print("Fetching recent messages...")
print("-" * 70)

try:
    # Get the last 5 messages
    messages = client.messages.list(limit=5)
    
    for msg in messages:
        print(f"Message SID: {msg.sid}")
        print(f"Status: {msg.status}")
        print(f"From: {msg.from_}")
        print(f"To: {msg.to}")
        print(f"Date: {msg.date_sent}")
        print(f"Error Code: {msg.error_code}")
        print(f"Error Message: {msg.error_message}")
        print(f"Body Preview: {msg.body[:50]}...")
        print("-" * 70)
        
except Exception as e:
    print(f"Error: {e}")
