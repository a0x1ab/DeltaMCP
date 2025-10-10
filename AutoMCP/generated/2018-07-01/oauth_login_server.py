# OAuth2 Login Flask Server (generated)
import os
import base64
import requests
from flask import Flask, request, redirect, session
from dotenv import load_dotenv
import secrets
import hashlib
import base64 as b64
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("OAUTH2_SECRET_KEY", secrets.token_hex(16))

print("[OAUTH LOGIN SERVER] Flask OAuth2 login server running.")
if __name__ == '__main__':
    app.run(debug=True, port=8888)
