from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

account_sid = os.getenv('TWILIO_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

print("Verified Phone Numbers on your Twilio account:")
print("-" * 50)

try:
    outgoing_caller_ids = client.outgoing_caller_ids.list()
    
    if not outgoing_caller_ids:
        print("No verified numbers found.")
        print("\nTo verify a number:")
        print("1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
        print("2. Click 'Add a new Caller ID'")
        print("3. Enter: +918501837598")
        print("4. Complete verification")
    else:
        for caller_id in outgoing_caller_ids:
            print(f"Phone Number: {caller_id.phone_number}")
            print(f"Friendly Name: {caller_id.friendly_name}")
            print("-" * 50)
            
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease verify your number manually at:")
    print("https://console.twilio.com/us1/develop/phone-numbers/manage/verified")
