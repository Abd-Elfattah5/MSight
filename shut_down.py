import os
import subprocess

def shutdown_server(port=8000):
    try:
        # Find the process using port 8000
        result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        for line in lines[1:]:  # Skip header
            if "LISTEN" in line:
                pid = int(line.split()[1])
                os.system(f"kill -9 {pid}")
                print(f"Server on port {port} (PID {pid}) terminated")
                return
        print(f"No server found on port {port}")
    except Exception as e:
        print(f"Error shutting down server: {str(e)}")

if __name__ == "__main__":
    shutdown_server()