import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.database.airflow_db import db

print("Checking DB Status locally...")
try:
    status = db.get_status()
    print("Status found:")
    for item in status:
        print(item)
except Exception as e:
    print(f"Error checking status: {e}")
