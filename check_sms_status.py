from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

# Check the status of your last message
message_sid = 'SMf22ec703112dbaa8366988ead2bf8fc7'  # Replace with your SID

try:
    message = client.messages(message_sid).fetch()
    print(f"Message SID: {message.sid}")
    print(f"Status: {message.status}")
    print(f"Error Code: {message.error_code}")
    print(f"Error Message: {message.error_message}")
    print(f"From: {message.from_}")
    print(f"To: {message.to}")
    print(f"Date Sent: {message.date_sent}")
except Exception as e:
    print(f"Error fetching message: {e}")
