import subprocess

def start_server():
    subprocess.Popen(["python", "server/server.py"])

def start_client():
    subprocess.Popen(["python", "client/client.py"])

if __name__ == "__main__":
    start_server()
    # Start multiple clients
    for _ in range(3):
        start_client()
