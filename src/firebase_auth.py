import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Environment variables
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY')
USER_EMAIL = os.getenv('USER_EMAIL')
USER_PASSWORD = os.getenv('USER_PASSWORD')
# BACKEND_ENDPOINT = os.getenv('BACKEND_ENDPOINT')

def get_firebase_id_token(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    data = {
        'email': email,
        'password': password,
        'returnSecureToken': True,
    }
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
    # print(response.json())
    return response.json()['idToken']

def make_authenticated_request(id_token):
    headers = {'Authorization': f'Bearer {id_token}'}
    response = requests.get("http://127.0.0.1:8000/hello-world/", headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", response.json())

if __name__ == '__main__':
    try:
        id_token = get_firebase_id_token(USER_EMAIL, USER_PASSWORD)
        print(id_token)
        # make_authenticated_request(id_token)
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")
