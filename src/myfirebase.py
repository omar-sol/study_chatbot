import json
from typing import Optional
from fastapi import Depends, HTTPException, status, Security, Query 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth, firestore

try:
    # Try to open the file and read the config
    with open('secure-config-store/student-chat-bot-firebase-adminsdk.json', 'r') as file:
        firebase_config = json.load(file)
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
except FileNotFoundError:
    print("Firebase config file not found")
except Exception as e:
    print(f"An error occurred while reading the Firebase config file: {e}")

oauth2_scheme = HTTPBearer()

# Define the get_current_user function with a change to handle unauthenticated requests
async def get_current_user(token: HTTPAuthorizationCredentials = Security(oauth2_scheme)) -> Optional[dict]:
    try:
        # Verify the token using Firebase Admin SDK
        decoded_token = auth.verify_id_token(token.credentials)
        return decoded_token
    except Exception:
        # Return None if authentication fails
        return None
    