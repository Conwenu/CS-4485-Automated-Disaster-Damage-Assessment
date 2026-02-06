cd Backend
python -m venv venv
venv\Scripts\activate
python -m uvicorn app.main:app --reload
Browser: http://127.0.0.1:8000/docs





