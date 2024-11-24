# run_server.py
import os
import sys

# Set the working directory to the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the server script
from server.server import main

if __name__ == "__main__":
    main()
