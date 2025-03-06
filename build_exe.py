import subprocess
import os
import webbrowser

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def open_browser(url):
    chrome_path=r"C:/Program Files/Google/Chrome/Application/chrome.exe"
    webbrowser.get(chrome_path).open(url)

def main():
    # Change directory to your Flask app directory
    os.chdir(r"/mnt/c/Users/santt/Desktop/CustomsPortal_App/fk_fraud_model")

    # Activate the virtual environment
    run_command("bash -c 'source ~/.fk_fraud_model/bin/activate && make run'")

    open_browser("http://localhost:5000/")

if __name__ == "__main__":
    main()
