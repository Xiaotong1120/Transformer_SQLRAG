import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL database connection
DB_PARAMS = {
    "dbname": "mydatabase",
    "user": "myuser",
    "password": "mypassword",
    "host": "localhost",
    "port": "5433"
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SQL_TIMEOUT = 30000

MAX_RESULTS_ROWS = 50000

SIMILARITY_THRESHOLD = 0.2

SIMILARITY_TOP_K =