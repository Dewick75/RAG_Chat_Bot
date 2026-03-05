import os
import sys

# Add the project directory to sys.path so app.py can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

mount_chainlit(app=app, target="app.py", path="/")
