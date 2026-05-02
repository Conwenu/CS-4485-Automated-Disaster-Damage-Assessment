cd Backend
python -m venv venv
venv\Scripts\activate.bat -> if u need to download it (pip install -r requirements.txt)
python -m uvicorn app.main:app --reload -> if not working ->(pip install uvicorn fastapi)
Browser: http://127.0.0.1:8000/docs

Backend is ready; call `POST /vlm/assess` with files `pre_image`, `post_image`, and `label_file`.
Use response fields `model.damage_level`, `model.confidence`, `model.reasoning`, `model.critical_regions`, `evaluation.ground_truth`, and `evaluation.match` in the UI.
For precomputed data, read `Backend/data/santa_rosa/results.json`.
