import subprocess

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "app/ui/streamlit_app.py"], check=False)
