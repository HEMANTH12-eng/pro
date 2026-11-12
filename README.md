# Safety Helmet Vision AI - Industrial Safety Helmet Detection System (Team 11)

An automated Machine Learning and Computer Vision web application built with **YOLOv8**, **OpenCV**, and **Streamlit** to detect whether industrial workers are wearing safety helmets (`helmet` vs `no_helmet`), display real-time counts of workers **NOT wearing helmets**, auto-start live webcam monitoring, and send instant **WhatsApp API notifications** to the manager's mobile.

## Team Members (Team 11)
- **S. Hemanth** (24691A3195)
- **G. Hemanth** (24691A3193)
- **P. Narasimhulu** (25695A3113)
- **Shaik Rehaman**

## Key Features
- **Live Webcam Auto-Start**: Immediately opens the live webcam stream on launch to detect helmet compliance in real time (with automatic fallback to Browser Camera on cloud servers).
- **Worker Violation Counter**: Displays live metrics for **Total Workers Detected**, **Wearing Helmet**, and **NOT Wearing Helmet**.
- **WhatsApp API Mobile Alerts**: Built-in REST API integration supporting **CallMeBot WhatsApp API**, **Meta Official WhatsApp Cloud API (v26.0)**, and **Twilio WhatsApp API** with configurable cooldown and a one-click test button.

## Project Structure
- `app.py` — Streamlit web application with Live Webcam auto-start, violation counters, and WhatsApp API notification integration.
- `helmet_detector_best.pt` — Trained YOLOv8n custom weights (`0: helmet`, `1: no_helmet`).
- `requirements.txt` — Python dependencies configured for local and cloud deployment.
- `packages.txt` — System-level Linux packages (`libgl1`, `libglib2.0-0`) for Streamlit Community Cloud.
- `Dockerfile` — Container configuration for Render, Railway, Google Cloud Run, or Hugging Face Spaces.
- `instructions_to_run_project.docx` — Step-by-step guide to run the project in VS Code.

## Run Locally (VS Code / Terminal)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows PowerShell:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   # Linux / macOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` in your browser.
