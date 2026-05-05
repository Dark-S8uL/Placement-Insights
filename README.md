# Smart Placement and Insights

This project is a complete AI/ML placement website built from the research paper, poster, and bundled dataset.

It now includes local login with `admin` and `student` roles, a sidebar-based dashboard, student profiles, analytics pages, and CSV download actions.

## Stack

- Frontend: React, HTML, CSS, and JavaScript
- Backend: FastAPI and Uvicorn
- ML: XGBoost model loaded with joblib, plus NumPy/Pandas for inference and dataset analysis
- Data: CSV placement dataset and generated model artifacts

## What the website includes

- Local login page with role-based access
- Placement prediction form
- Dataset-wide analytics and group trends
- Model metadata and feature transparency
- Role suggestions, readiness scoring, and next steps
- Admin job postings that surface eligible links on student profiles
- Research summary aligned with the poster and paper
- React-powered routed frontend served through FastAPI
- Admin and student profile views
- CSV downloads for overall statistics, per-student data, raw dataset, and predictions

## Run locally on Windows

The simplest way to start the app from a fresh clone is:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_local.ps1
```

If you prefer to run the steps manually:

1. Open a terminal in the repository root.
2. Create a virtual environment:

```powershell
py -3 -m venv .venv
```

3. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install the project dependencies:

```powershell
pip install -r requirements.txt
```

5. Start the application:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

6. Open the app in your browser:

```text
http://127.0.0.1:8000
```

7. Log in with one of the seeded local accounts:

```text
Admin: admin / admin123
Student: student1 / student123
Student: student2 / student123
Student: student3 / student123
```

8. Use the sidebar to switch between Dashboard, Analytics, Students, Profiles, and Downloads.

## Notes

- The saved model is bundled in `output/model.pkl` and the dataset is bundled in `output/final_dataset.csv`.
- The research notebooks and original dataset files are bundled in `Dataset/` for reference and offline analysis.
- Local login credentials are stored in `data/users.json`.
  On first run the app will migrate `data/users.json` into a local SQLite database file `db.sqlite3` (created in the project root).
- The repo is fully self-contained for local use; no external model download or preprocessing step is required.
- If the machine has limited disk space, install into a clean environment or a drive with enough free space before running the server.
- The server exposes the prediction API at `/api/predict` and the metadata API at `/api/model-info`.
- New: user accounts are now persisted in a local SQLite DB by default. To override the DB path (or use Postgres), set the `DATABASE_URL` environment variable before starting the server.
  Example (default SQLite): `DATABASE_URL=sqlite:///./db.sqlite3`
