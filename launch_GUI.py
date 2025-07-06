import subprocess
import sys
import os
from addmo.util.load_save_utils import root_dir

def launch_streamlit():
    app_path = os.path.join(root_dir(), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    launch_streamlit()