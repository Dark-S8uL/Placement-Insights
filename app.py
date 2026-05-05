from collections import Counter, defaultdict
from csv import DictReader, DictWriter, reader as CsvReader
from datetime import date, datetime
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

import joblib
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import os
import sqlite3

# database / auth
from passlib.context import CryptContext


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "model.pkl"
DATASET_PATH = BASE_DIR / "output" / "final_dataset.csv"
USERS_PATH = BASE_DIR / "data" / "users.json"
FEATURE_NAMES = [
    "CGPA",
    "Internships",
    "Projects",
    "Certifications",
    "Communication_Skills",
    "Aptitude_Score",
    "Backlogs",
]

PROJECT_OVERVIEW = {
    "title": "Smart Placement and Insights",
    "summary": (
        "A placement analytics website that combines a trained machine learning model "
        "with dataset-level insights from the research paper and poster."
    ),
    "objectives": [
        "Predict whether a student is likely to be placed",
        "Surface the strongest placement drivers from the dataset",
        "Provide skill-gap guidance and role suggestions",
        "Package the research into a clean, interactive website",
    ],
    "tech_stack": {
        "frontend": ["React", "HTML", "CSS", "JavaScript"],
        "backend": ["FastAPI", "Uvicorn"],
        "ml": ["scikit-learn", "XGBoost", "joblib", "NumPy"],
        "data": ["CSV dataset", "placement research notebooks"],
    },
    "features": [
        "Placement prediction form",
        "Dataset-wide placement analytics",
        "Model confidence and readiness score",
        "Role recommendations and next steps",
        "Admin job postings with student eligibility filters",
        "React-powered responsive single-page interface",
    ],
}


app = FastAPI(title="Smart Placement and Insights", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SESSIONS: Dict[str, Dict[str, Any]] = {}


pwd_context = CryptContext(schemes=["pbkdf2_sha256","bcrypt"], deprecated="auto")


DB_PATH = BASE_DIR / "db.sqlite3"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            display_name TEXT,
            student_id INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_postings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_name TEXT NOT NULL,
            job_link TEXT NOT NULL,
            min_cgpa REAL NOT NULL,
            max_backlogs REAL NOT NULL,
            created_by TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER UNIQUE,
            resume_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


def db_get_user_by_username(username: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,username,password_hash,role,display_name,student_id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "role": row[3],
        "display_name": row[4],
        "student_id": row[5],
    }


def db_list_users():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,username,role,display_name,student_id FROM users ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    users = []
    for r in rows:
        users.append({"id": r[0], "username": r[1], "role": r[2], "display_name": r[3], "student_id": r[4]})
    return users


def db_student_id_exists(student_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE role = 'student' AND student_id = ? LIMIT 1", (student_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def db_student_id_link_count(student_id: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'student' AND student_id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    return int(row[0] if row else 0)


def db_create_user(username: str, password: str, role: str = "student", display_name: Optional[str] = None, student_id: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username,password_hash,role,display_name,student_id) VALUES (?,?,?,?,?)",
            (username, hash_password(password), role, display_name, student_id),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return None
    conn.close()
    return db_get_user_by_username(username)


def db_list_job_postings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id,form_name,job_link,min_cgpa,max_backlogs,created_by,created_at FROM job_postings ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    postings = []
    for row in rows:
        postings.append(
            {
                "id": row[0],
                "form_name": row[1],
                "job_link": row[2],
                "min_cgpa": row[3],
                "max_backlogs": row[4],
                "created_by": row[5],
                "created_at": row[6],
            }
        )
    return postings


def db_create_job_posting(form_name: str, job_link: str, min_cgpa: float, max_backlogs: float, created_by: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO job_postings (form_name,job_link,min_cgpa,max_backlogs,created_by) VALUES (?,?,?,?,?)",
        (form_name, job_link, float(min_cgpa), float(max_backlogs), created_by),
    )
    conn.commit()
    posting_id = cur.lastrowid
    conn.close()
    for posting in db_list_job_postings():
        if posting["id"] == posting_id:
            return posting
    return None


def db_get_user_by_student_id(student_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id,username,role,display_name,student_id FROM users WHERE role = 'student' AND student_id = ? LIMIT 1",
        (student_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "username": row[1],
        "role": row[2],
        "display_name": row[3],
        "student_id": row[4],
    }


def db_get_resume(student_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT resume_json FROM resume_data WHERE student_id = ? LIMIT 1",
        (student_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def db_upsert_resume(student_id: int, resume_payload: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    resume_json = json.dumps(resume_payload, ensure_ascii=False)
    cur.execute(
        """
        INSERT INTO resume_data (student_id, resume_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(student_id)
        DO UPDATE SET resume_json = excluded.resume_json, updated_at = CURRENT_TIMESTAMP
        """,
        (student_id, resume_json),
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()
    init_db()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


class FallbackModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        row = np.asarray(features, dtype=float)[0]
        cgpa, internships, projects, certifications, communication, aptitude, backlogs = row
        score = (
            cgpa * 11.0
            + internships * 8.0
            + projects * 6.0
            + certifications * 4.0
            + communication * 0.18
            + aptitude * 0.22
            - backlogs * 10.0
        )
        return np.array([1 if score >= 100 else 0])

    def predict_proba(self, features):
        row = np.asarray(features, dtype=float)[0]
        cgpa, internships, projects, certifications, communication, aptitude, backlogs = row
        score = (
            cgpa * 11.0
            + internships * 8.0
            + projects * 6.0
            + certifications * 4.0
            + communication * 0.18
            + aptitude * 0.22
            - backlogs * 10.0
        )
        probability = 1.0 / (1.0 + np.exp(-(score - 100.0) / 10.0))
        return np.array([[1.0 - probability, probability]])


def load_rows():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(DictReader(handle))




def normalize_student_id(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_student_index(rows):
    student_index = {}
    for row in rows:
        student_id = normalize_student_id(row.get("student_id") or row.get("StudentID"))
        if student_id is not None:
            student_index[student_id] = row
    return student_index


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_placed(value):
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "placed", "pass", "y"}


def average(values):
    values = [value for value in values if value is not None]
    return round(mean(values), 2) if values else 0.0


def build_insights(rows, branch_min_students=500):
    total_students = len(rows)
    placement_counts = Counter()
    branch_stats = defaultdict(lambda: {"total": 0, "placed": 0})
    tier_stats = defaultdict(lambda: {"total": 0, "placed": 0})
    numeric_groups = defaultdict(lambda: {"placed": [], "not_placed": []})

    numeric_columns = [
        "CGPA",
        "Internships",
        "Projects",
        "Certifications",
        "Programming_Skills",
        "Aptitude_Score",
        "Communication_Skills",
        "logical_reasoning_score",
        "Hackathons",
        "github_repos",
        "linkedin_connections",
        "mock_interview_score",
        "Attendance",
        "Backlogs",
        "extracurricular_score",
        "Leadership",
        "volunteer_experience",
        "sleep_hours",
        "study_hours_per_day",
        "10th_Percentage",
        "12th_Percentage",
        "DSA_Score",
        "Teamwork",
        "Skill_Score",
        "Experience_Score",
        "Academic_Consistency",
    ]

    for row in rows:
        placed = is_placed(row.get("Placement", row.get("PlacementStatus", "")))
        status_key = "Placed" if placed else "Not Placed"
        placement_counts[status_key] += 1

        branch = row.get("branch") or row.get("Core_Subjects") or "Unknown"
        tier = row.get("college_tier") or row.get("company_type") or "Unknown"
        branch_stats[branch]["total"] += 1
        tier_stats[tier]["total"] += 1
        if placed:
            branch_stats[branch]["placed"] += 1
            tier_stats[tier]["placed"] += 1

        for column in numeric_columns:
            value = to_float(row.get(column))
            if value is None:
                continue
            numeric_groups[column]["placed" if placed else "not_placed"].append(value)

    total_placed = placement_counts["Placed"]
    total_not_placed = placement_counts["Not Placed"]
    placement_rate = round((total_placed / total_students) * 100, 2) if total_students else 0.0

    branch_rows = []
    for branch, stats in branch_stats.items():
        if stats["total"] < branch_min_students:
            continue
        branch_rows.append(
            {
                "label": branch,
                "placement_rate": round((stats["placed"] / stats["total"]) * 100, 2),
                "students": stats["total"],
            }
        )
    branch_rows.sort(key=lambda item: item["placement_rate"], reverse=True)

    tier_rows = [
        {
            "label": tier,
            "placement_rate": round((stats["placed"] / stats["total"]) * 100, 2),
            "students": stats["total"],
        }
        for tier, stats in tier_stats.items()
        if stats["total"]
    ]
    tier_rows.sort(key=lambda item: item["placement_rate"], reverse=True)

    driver_rows = []
    for column, groups in numeric_groups.items():
        placed_values = groups["placed"]
        not_placed_values = groups["not_placed"]
        if len(placed_values) < 10 or len(not_placed_values) < 10:
            continue
        placed_avg = average(placed_values)
        not_placed_avg = average(not_placed_values)
        delta = round(placed_avg - not_placed_avg, 2)
        driver_rows.append(
            {
                "label": column,
                "placed_avg": placed_avg,
                "not_placed_avg": not_placed_avg,
                "delta": delta,
            }
        )

    driver_rows.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "summary": {
            "total_students": total_students,
            "placed_students": total_placed,
            "not_placed_students": total_not_placed,
            "placement_rate": placement_rate,
            "avg_cgpa": average(numeric_groups["CGPA"]["placed"] + numeric_groups["CGPA"]["not_placed"]),
            "avg_aptitude": average(numeric_groups["Aptitude_Score"]["placed"] + numeric_groups["Aptitude_Score"]["not_placed"]),
            "avg_communication": average(numeric_groups["Communication_Skills"]["placed"] + numeric_groups["Communication_Skills"]["not_placed"]),
            "avg_projects": average(numeric_groups["Projects"]["placed"] + numeric_groups["Projects"]["not_placed"]),
            "avg_internships": average(numeric_groups["Internships"]["placed"] + numeric_groups["Internships"]["not_placed"]),
        },
        "top_branches": branch_rows[:6],
        "college_tiers": tier_rows,
        "key_drivers": driver_rows[:6],
    }


def readiness_score(data):
    score = (
        min(data.CGPA, 10.0) * 8.0
        + min(data.Internships, 6.0) * 7.0
        + min(data.Projects, 8.0) * 6.0
        + min(data.Certifications, 8.0) * 4.5
        + min(data.Communication_Skills, 100.0) * 0.2
        + min(data.Aptitude_Score, 100.0) * 0.22
        - min(data.Backlogs, 10.0) * 8.0
    )
    return round(max(0.0, min(100.0, score)), 1)


def role_recommendation(data):
    signals = []
    if data.CGPA >= 8.2 and data.Projects >= 3:
        signals.append("Software development")
    if data.Aptitude_Score >= 80 and data.Communication_Skills >= 70:
        signals.append("Business analyst")
    if data.Internships >= 2 and data.Certifications >= 2:
        signals.append("Industry-ready internship track")
    if data.Backlogs > 0:
        signals.append("Priority: clear backlogs")
    if not signals:
        signals.append("Foundational placement support")
    return signals


def build_model_info(model):
    algorithm = type(model).__name__
    info = {
        "algorithm": algorithm,
        "supports_probability": hasattr(model, "predict_proba"),
        "feature_names": FEATURE_NAMES,
        "target": "Placement",
        "input_size": len(FEATURE_NAMES),
        "notes": [
            "The model consumes the same seven core features shown on the website.",
            "Dataset analytics are computed from the bundled placement data file.",
        ],
    }

    if hasattr(model, "feature_importances_"):
        importances = list(getattr(model, "feature_importances_"))
        ranked = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda item: item[1],
            reverse=True,
        )
        info["top_importances"] = [
            {"label": name, "score": round(float(score) * 100, 2)}
            for name, score in ranked[:5]
        ]
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"))[0]
        ranked = sorted(
            zip(FEATURE_NAMES, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        info["top_importances"] = [
            {"label": name, "score": round(float(score), 4)}
            for name, score in ranked[:5]
        ]
    else:
        info["top_importances"] = []

    return info


def get_user_public(user):
    # supports both dict-like and SQLModel objects
    if isinstance(user, dict):
        return {
            "username": user["username"],
            "role": user["role"],
            "display_name": user.get("display_name") or user["username"],
            "student_id": user.get("student_id"),
        }
    else:
        return {
            "username": getattr(user, "username", None),
            "role": getattr(user, "role", None),
            "display_name": getattr(user, "display_name", None) or getattr(user, "username", None),
            "student_id": getattr(user, "student_id", None),
        }


def get_user_by_username(username: str):
    return db_get_user_by_username(username)


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.split(" ", 1)[1].strip()
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session


def get_current_user_optional(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return SESSIONS.get(token)


def require_admin(user = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def get_row_for_student(student_id):
    return STUDENT_INDEX.get(normalize_student_id(student_id))


def dataset_fieldnames():
    if DATASET_PATH.exists():
        with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = DictReader(handle)
            if reader.fieldnames:
                return list(reader.fieldnames)

    fieldnames = []
    seen = set()
    for row in ROWS:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def persist_rows_to_dataset():
    fieldnames = dataset_fieldnames()
    if not fieldnames:
        return

    with DATASET_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_student_row(row):
    normalized = dict(row)
    normalized["student_id"] = normalize_student_id(row.get("student_id") or row.get("StudentID"))
    return normalized


def job_is_eligible(row, posting):
    cgpa = to_float(row.get("CGPA"))
    backlogs = to_float(row.get("Backlogs"))
    if cgpa is None or backlogs is None:
        return False
    return cgpa >= float(posting.get("min_cgpa") or 0) and backlogs <= float(posting.get("max_backlogs") or 0)


def job_postings_for_student(row):
    return [posting for posting in db_list_job_postings() if job_is_eligible(row, posting)]


def update_student_record(student_id, updates):
    global insights

    target_id = normalize_student_id(student_id)
    if target_id is None:
        raise HTTPException(status_code=400, detail="Student ID is required")

    row = get_row_for_student(target_id)
    updated_row = dict(row) if row else {"student_id": target_id}
    for key, value in updates.items():
        if value is not None and value != "":
            updated_row[key] = value

    if not updated_row.get("branch"):
        updated_row["branch"] = "Other"
    if not updated_row.get("college_tier"):
        updated_row["college_tier"] = "Tier 2"

    if row is None:
        ROWS.append(updated_row)
    else:
        for index, existing in enumerate(ROWS):
            if normalize_student_id(existing.get("student_id") or existing.get("StudentID")) == target_id:
                ROWS[index] = updated_row
                break

    updated_row = normalize_student_row(updated_row)

    STUDENT_INDEX[target_id] = updated_row
    persist_rows_to_dataset()
    insights = build_insights(ROWS)
    return build_student_profile(updated_row)


def csv_response(filename, csv_text):
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def core_vector_from_values(cgpa, internships, projects, certifications, communication_skills, aptitude_score, backlogs):
    return np.array(
        [[
            float(cgpa),
            float(internships),
            float(projects),
            float(certifications),
            float(communication_skills),
            float(aptitude_score),
            float(backlogs),
        ]],
        dtype=float,
    )


def model_predict_from_vector(vector):
    features = vector
    try:
        prediction = model.predict(features)[0]
    except ValueError as ve:
        message = str(ve)
        import re

        match = re.search(r"expected:\s*(\d+),\s*got\s*(\d+)", message)
        if not match:
            raise ve
        expected = int(match.group(1))
        got = int(match.group(2))
        if got >= expected:
            raise ve
        padded = np.concatenate([features, np.zeros((1, expected - got), dtype=float)], axis=1)
        prediction = model.predict(padded)[0]
        features = padded

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            proba_values = model.predict_proba(features)[0]
            positive_index = 1
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                for candidate in (1, "1", True, "Placed", "placed", "Yes", "yes"):
                    if candidate in classes:
                        positive_index = classes.index(candidate)
                        break
            probability = float(proba_values[positive_index])
        except Exception:
            probability = None

    if isinstance(prediction, str):
        result_label = "Placed" if prediction.strip().lower() in {"1", "true", "yes", "placed"} else "Not Placed"
    else:
        result_label = "Placed" if int(prediction) == 1 else "Not Placed"

    return result_label, probability


def predict_row(row):
    vector = core_vector_from_values(
        to_float(row.get("CGPA")) or 0.0,
        to_float(row.get("Internships")) or 0.0,
        to_float(row.get("Projects")) or 0.0,
        to_float(row.get("Certifications")) or 0.0,
        to_float(row.get("Communication_Skills")) or 0.0,
        to_float(row.get("Aptitude_Score")) or 0.0,
        to_float(row.get("Backlogs")) or 0.0,
    )
    return model_predict_from_vector(vector)


def student_row_from_values(values):
    result_label, _ = model_predict_from_vector(
        core_vector_from_values(
            values["CGPA"],
            values["Internships"],
            values["Projects"],
            values["Certifications"],
            values["Communication_Skills"],
            values["Aptitude_Score"],
            values["Backlogs"],
        )
    )
    return {
        "student_id": values["student_id"],
        "branch": values["branch"],
        "college_tier": values["college_tier"],
        "CGPA": values["CGPA"],
        "Internships": values["Internships"],
        "Projects": values["Projects"],
        "Certifications": values["Certifications"],
        "Communication_Skills": values["Communication_Skills"],
        "Aptitude_Score": values["Aptitude_Score"],
        "Backlogs": values["Backlogs"],
        "Placement": "1" if result_label == "Placed" else "0",
    }


def parse_bulk_student_row(row, line_number):
    missing = [field for field in BULK_STUDENT_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Line {line_number}: missing required field(s): {', '.join(missing)}")

    parsed = {
        "username": str(row["username"]).strip(),
        "password": str(row["password"]),
        "display_name": str(row["display_name"]).strip(),
        "branch": str(row["branch"]).strip(),
        "college_tier": str(row["college_tier"]).strip(),
    }
    if not parsed["username"]:
        raise ValueError(f"Line {line_number}: username is required")
    if not parsed["password"]:
        raise ValueError(f"Line {line_number}: password is required")

    student_id = normalize_student_id(row.get("student_id"))
    if student_id is None:
        raise ValueError(f"Line {line_number}: student_id must be a number")
    parsed["student_id"] = student_id

    for field in [
        "CGPA",
        "Internships",
        "Projects",
        "Certifications",
        "Communication_Skills",
        "Aptitude_Score",
        "Backlogs",
    ]:
        value = to_float(row.get(field))
        if value is None:
            raise ValueError(f"Line {line_number}: {field} must be a number")
        parsed[field] = value

    if not 0 <= parsed["CGPA"] <= 10:
        raise ValueError(f"Line {line_number}: CGPA must be between 0 and 10")
    if not 0 <= parsed["Communication_Skills"] <= 100:
        raise ValueError(f"Line {line_number}: Communication_Skills must be between 0 and 100")
    if not 0 <= parsed["Aptitude_Score"] <= 100:
        raise ValueError(f"Line {line_number}: Aptitude_Score must be between 0 and 100")
    for field in ["Internships", "Projects", "Certifications", "Backlogs"]:
        if parsed[field] < 0:
            raise ValueError(f"Line {line_number}: {field} cannot be negative")

    return parsed


def normalize_bulk_csv_text(text):
    csv_rows = list(CsvReader(StringIO(text)))
    if not csv_rows:
        return text
    first_row = csv_rows[0]
    if len(first_row) == 1 and "," in first_row[0]:
        return "\n".join(row[0] for row in csv_rows if row)
    return text


def add_student_row(row):
    ROWS.append(row)
    STUDENT_INDEX[row["student_id"]] = row


def build_student_profile(row):
    result_label, probability = predict_row(row)
    student_id = normalize_student_id(row.get("student_id") or row.get("StudentID"))
    eligible_jobs = job_postings_for_student(row)
    return {
        "student_id": student_id,
        "branch": row.get("branch") or row.get("Core_Subjects"),
        "college_tier": row.get("college_tier") or row.get("company_type"),
        "placement": result_label,
        "probability": round(probability * 100, 2) if probability is not None else None,
        "eligible_jobs": eligible_jobs,
        "eligible_job_count": len(eligible_jobs),
        "readiness_score": round(
            min(to_float(row.get("CGPA")) or 0, 10.0) * 8.0
            + min(to_float(row.get("Internships")) or 0, 6.0) * 7.0
            + min(to_float(row.get("Projects")) or 0, 8.0) * 6.0
            + min(to_float(row.get("Certifications")) or 0, 8.0) * 4.5
            + min(to_float(row.get("Communication_Skills")) or 0, 100.0) * 0.2
            + min(to_float(row.get("Aptitude_Score")) or 0, 100.0) * 0.22
            - min(to_float(row.get("Backlogs")) or 0, 10.0) * 8.0,
            1,
        ),
        "profile": {
            "CGPA": to_float(row.get("CGPA")),
            "Internships": to_float(row.get("Internships")),
            "Projects": to_float(row.get("Projects")),
            "Certifications": to_float(row.get("Certifications")),
            "Communication_Skills": to_float(row.get("Communication_Skills")),
            "Aptitude_Score": to_float(row.get("Aptitude_Score")),
            "Backlogs": to_float(row.get("Backlogs")),
        },
    }


def student_department(row):
    for key in ("branch", "Core_Subjects"):
        value = str(row.get(key) or "").strip()
        if value and any(character.isalpha() for character in value):
            return value
    return "Unknown"


def available_departments():
    departments = {department for department in (student_department(row) for row in ROWS) if department != "Unknown"}
    return sorted(departments, key=lambda value: value.lower())


def student_tier(row):
    for key in ("college_tier", "company_type"):
        raw_value = str(row.get(key) or "").strip()
        if not raw_value:
            continue
        if any(character.isalpha() for character in raw_value):
            return raw_value
        numeric_value = to_float(raw_value)
        if numeric_value is not None and numeric_value in {1.0, 2.0, 3.0}:
            return f"Tier {int(numeric_value)}"
    return "Unknown"


def available_tiers():
    tiers = {student_tier(row) for row in ROWS if student_tier(row) != "Unknown"}
    return sorted(tiers, key=lambda value: value.lower())


def parse_date_value(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    for date_format in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, date_format).date()
        except ValueError:
            continue
    return None


def row_date(row):
    preferred_keys = [
        "placement_date",
        "date",
        "created_at",
        "created_on",
        "updated_at",
        "timestamp",
    ]
    for key in preferred_keys:
        parsed = parse_date_value(row.get(key))
        if parsed is not None:
            return parsed
    for key, value in row.items():
        if "date" not in str(key).lower():
            continue
        parsed = parse_date_value(value)
        if parsed is not None:
            return parsed
    return None


def has_date_filter_support(rows):
    for row in rows:
        if row_date(row) is not None:
            return True
    return False


def filtered_analytics_rows(rows, department=None, tier=None, start_date=None, end_date=None):
    date_supported = has_date_filter_support(rows)
    try:
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if (start or end) and not date_supported:
        raise HTTPException(status_code=400, detail="Date filter is not available for the current dataset.")
    if start and end and end < start:
        raise HTTPException(status_code=400, detail="End date must be on or after start date.")

    selected = []
    for row in rows:
        if department and student_department(row).lower() != department.lower():
            continue
        if tier and student_tier(row).lower() != tier.lower():
            continue
        if start or end:
            current_date = row_date(row)
            if current_date is None:
                continue
            if start and current_date < start:
                continue
            if end and current_date > end:
                continue
        selected.append(row)

    return selected, date_supported


def ranked_student_rows(department):
    selected_department = department.strip()
    rankings = []
    for row in ROWS:
        if student_department(row).lower() != selected_department.lower():
            continue
        result_label, probability = predict_row(row)
        probability_percent = round(probability * 100, 2) if probability is not None else None
        rankings.append(
            {
                "student_id": normalize_student_id(row.get("student_id") or row.get("StudentID")),
                "department": student_department(row),
                "college_tier": row.get("college_tier") or row.get("company_type") or "Unknown",
                "CGPA": to_float(row.get("CGPA")),
                "Internships": to_float(row.get("Internships")),
                "Projects": to_float(row.get("Projects")),
                "Certifications": to_float(row.get("Certifications")),
                "Communication_Skills": to_float(row.get("Communication_Skills")),
                "Aptitude_Score": to_float(row.get("Aptitude_Score")),
                "Backlogs": to_float(row.get("Backlogs")),
                "prediction": result_label,
                "probability": probability_percent,
            }
        )

    rankings.sort(
        key=lambda item: (
            item["probability"] is not None,
            item["probability"] if item["probability"] is not None else -1,
        ),
        reverse=True,
    )
    for index, item in enumerate(rankings, start=1):
        item["rank"] = index
    return rankings


def student_rankings_csv(department):
    rows = ranked_student_rows(department)
    output = StringIO()
    fieldnames = [
        "rank",
        "student_id",
        "department",
        "college_tier",
        "prediction",
        "probability",
        "CGPA",
        "Internships",
        "Projects",
        "Certifications",
        "Communication_Skills",
        "Aptitude_Score",
        "Backlogs",
    ]
    writer = DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output.getvalue()


def report_rows_for_scope(scope: str, department: Optional[str] = None):
    normalized_scope = (scope or "overall").strip().lower()
    selected_department = (department or "").strip()
    if normalized_scope not in {"overall", "department"}:
        raise HTTPException(status_code=400, detail="scope must be either 'overall' or 'department'")
    if normalized_scope == "department" and not selected_department:
        raise HTTPException(status_code=400, detail="department is required when scope is 'department'")

    rows = ROWS if normalized_scope == "overall" else [
        row for row in ROWS if student_department(row).lower() == selected_department.lower()
    ]
    if not rows:
        raise HTTPException(status_code=404, detail="No data found for the selected report filter.")
    return rows, normalized_scope, selected_department


def improvement_report_csv(scope: str = "overall", department: Optional[str] = None):
    rows, normalized_scope, selected_department = report_rows_for_scope(scope, department)
    output = StringIO()
    fieldnames = [
        "scope",
        "department",
        "feature",
        "overall_avg",
        "placed_avg",
        "not_placed_avg",
        "gap",
        "priority",
        "guidance",
    ]
    writer = DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    features = [
        ("CGPA", "higher"),
        ("Internships", "higher"),
        ("Projects", "higher"),
        ("Certifications", "higher"),
        ("Communication_Skills", "higher"),
        ("Aptitude_Score", "higher"),
        ("Backlogs", "lower"),
    ]

    for feature, direction in features:
        placed_values = []
        not_placed_values = []
        all_values = []
        for row in rows:
            value = to_float(row.get(feature))
            if value is None:
                continue
            all_values.append(value)
            if is_placed(row.get("Placement", row.get("PlacementStatus", ""))):
                placed_values.append(value)
            else:
                not_placed_values.append(value)

        if not all_values:
            continue

        overall_avg = average(all_values)
        placed_avg = average(placed_values)
        not_placed_avg = average(not_placed_values)
        gap = round(placed_avg - not_placed_avg, 2)

        if direction == "higher":
            priority = "High" if gap >= 0.5 else "Medium" if gap >= 0.1 else "Low"
            guidance = f"Raise {feature.replace('_', ' ')} toward placed average ({placed_avg})."
        else:
            priority = "High" if gap <= -0.5 else "Medium" if gap < 0 else "Low"
            guidance = f"Keep {feature.replace('_', ' ')} at or below placed average ({placed_avg})."

        writer.writerow(
            {
                "scope": normalized_scope,
                "department": selected_department if normalized_scope == "department" else "All",
                "feature": feature,
                "overall_avg": overall_avg,
                "placed_avg": placed_avg,
                "not_placed_avg": not_placed_avg,
                "gap": gap,
                "priority": priority,
                "guidance": guidance,
            }
        )
    return output.getvalue()


def academic_report_csv(scope: str = "overall", department: Optional[str] = None):
    rows, normalized_scope, selected_department = report_rows_for_scope(scope, department)
    output = StringIO()
    fieldnames = [
        "scope",
        "department",
        "total_students",
        "placed_students",
        "not_placed_students",
        "placement_rate",
        "avg_cgpa",
        "avg_aptitude",
        "avg_communication",
        "avg_projects",
        "avg_internships",
        "avg_backlogs",
    ]
    writer = DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    grouped_rows = defaultdict(list)
    if normalized_scope == "overall":
        for row in rows:
            grouped_rows[student_department(row)].append(row)
    else:
        grouped_rows[selected_department] = rows

    for group_name, group_rows in sorted(grouped_rows.items(), key=lambda item: item[0].lower()):
        placed_count = 0
        cgpa_values = []
        aptitude_values = []
        communication_values = []
        projects_values = []
        internships_values = []
        backlogs_values = []

        for row in group_rows:
            if is_placed(row.get("Placement", row.get("PlacementStatus", ""))):
                placed_count += 1
            value = to_float(row.get("CGPA"))
            if value is not None:
                cgpa_values.append(value)
            value = to_float(row.get("Aptitude_Score"))
            if value is not None:
                aptitude_values.append(value)
            value = to_float(row.get("Communication_Skills"))
            if value is not None:
                communication_values.append(value)
            value = to_float(row.get("Projects"))
            if value is not None:
                projects_values.append(value)
            value = to_float(row.get("Internships"))
            if value is not None:
                internships_values.append(value)
            value = to_float(row.get("Backlogs"))
            if value is not None:
                backlogs_values.append(value)

        total_students = len(group_rows)
        not_placed = total_students - placed_count
        placement_rate = round((placed_count / total_students) * 100, 2) if total_students else 0.0

        writer.writerow(
            {
                "scope": normalized_scope,
                "department": group_name if normalized_scope == "overall" else selected_department,
                "total_students": total_students,
                "placed_students": placed_count,
                "not_placed_students": not_placed,
                "placement_rate": placement_rate,
                "avg_cgpa": average(cgpa_values),
                "avg_aptitude": average(aptitude_values),
                "avg_communication": average(communication_values),
                "avg_projects": average(projects_values),
                "avg_internships": average(internships_values),
                "avg_backlogs": average(backlogs_values),
            }
        )
    return output.getvalue()


def overall_stats_csv():
    buffer = StringIO()
    writer = DictWriter(buffer, fieldnames=["section", "label", "metric", "value"])
    writer.writeheader()
    summary = insights["summary"]
    for key, value in summary.items():
        writer.writerow({"section": "summary", "label": "", "metric": key, "value": value})
    for item in insights["top_branches"]:
        writer.writerow({"section": "top_branch", "label": item["label"], "metric": "placement_rate", "value": item["placement_rate"]})
    for item in insights["college_tiers"]:
        writer.writerow({"section": "college_tier", "label": item["label"], "metric": "placement_rate", "value": item["placement_rate"]})
    for item in insights["key_drivers"]:
        writer.writerow({"section": "key_driver", "label": item["label"], "metric": "delta", "value": item["delta"]})
    return buffer.getvalue()


def student_csv(row):
    buffer = StringIO()
    profile = build_student_profile(row)
    fieldnames = ["student_id", "branch", "college_tier", "placement", "probability", "readiness_score", "CGPA", "Internships", "Projects", "Certifications", "Communication_Skills", "Aptitude_Score", "Backlogs"]
    writer = DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "student_id": profile["student_id"],
        "branch": profile["branch"],
        "college_tier": profile["college_tier"],
        "placement": profile["placement"],
        "probability": profile["probability"],
        "readiness_score": profile["readiness_score"],
        **profile["profile"],
    })
    return buffer.getvalue()


def dataset_predictions_csv():
    df = pd.DataFrame(ROWS)
    features = df[["CGPA", "Internships", "Projects", "Certifications", "Communication_Skills", "Aptitude_Score", "Backlogs"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = np.hstack([features.to_numpy(dtype=float), np.zeros((len(features), max(0, getattr(model, "n_features_in_", 7) - 7)), dtype=float)])
    preds = model.predict(matrix)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(matrix)
        except Exception:
            probabilities = None
    output = StringIO()
    writer = DictWriter(output, fieldnames=["student_id", "prediction", "probability", "CGPA", "Internships", "Projects", "Certifications", "Communication_Skills", "Aptitude_Score", "Backlogs"])
    writer.writeheader()
    for index, row in df.iterrows():
        probability = None
        if probabilities is not None:
            probability = float(probabilities[index][1]) if probabilities.shape[1] > 1 else float(probabilities[index][0])
        writer.writerow({
            "student_id": row.get("student_id"),
            "prediction": preds[index],
            "probability": probability,
            "CGPA": row.get("CGPA"),
            "Internships": row.get("Internships"),
            "Projects": row.get("Projects"),
            "Certifications": row.get("Certifications"),
            "Communication_Skills": row.get("Communication_Skills"),
            "Aptitude_Score": row.get("Aptitude_Score"),
            "Backlogs": row.get("Backlogs"),
        })
    return output.getvalue()


RESUME_FIELDS = [
    "full_name",
    "headline",
    "email",
    "phone",
    "location",
    "linkedin",
    "leetcode",
    "github",
    "summary",
    "education",
    "education_1",
    "education_2",
    "education_3",
    "experience",
    "experience_1_title",
    "experience_1_meta",
    "experience_1_points",
    "projects",
    "project_1_title",
    "project_1_link",
    "project_1_points",
    "project_2_title",
    "project_2_link",
    "project_2_points",
    "skills",
    "skills_programming",
    "skills_frontend",
    "skills_backend",
    "skills_databases",
    "skills_tools",
    "certifications",
    "achievements",
]


def default_resume_payload(display_name: Optional[str] = None):
    return {
        "full_name": display_name or "",
        "headline": "",
        "email": "",
        "phone": "",
        "location": "",
        "linkedin": "",
        "leetcode": "",
        "github": "",
        "summary": "",
        "education": "",
        "education_1": "",
        "education_2": "",
        "education_3": "",
        "experience": "",
        "experience_1_title": "",
        "experience_1_meta": "",
        "experience_1_points": "",
        "projects": "",
        "project_1_title": "",
        "project_1_link": "",
        "project_1_points": "",
        "project_2_title": "",
        "project_2_link": "",
        "project_2_points": "",
        "skills": "",
        "skills_programming": "",
        "skills_frontend": "",
        "skills_backend": "",
        "skills_databases": "",
        "skills_tools": "",
        "certifications": "",
        "achievements": "",
    }


def normalize_resume_payload(payload: Dict[str, Any], display_name: Optional[str] = None):
    normalized = default_resume_payload(display_name)
    for key in RESUME_FIELDS:
        value = payload.get(key)
        normalized[key] = str(value).strip() if value is not None else ""
    if not normalized["full_name"] and display_name:
        normalized["full_name"] = display_name
    return normalized


def _extract_first_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def polish_resume_with_gemini(resume_payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return resume_payload

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={urllib_parse.quote(api_key)}"
    )

    prompt = {
        "instruction": (
            "Rewrite this student resume content into concise, professional, ATS-friendly text. "
            "Keep facts truthful to source content, improve alignment, grammar, and bullet structure, "
            "and do not invent employers, dates, scores, or certifications."
        ),
        "required_output": {
            "format": "JSON object only",
            "fields": RESUME_FIELDS,
            "rules": [
                "Return every field as a string.",
                "Preserve full_name, email, phone, location, linkedin, github as factual contact fields.",
                "Use clear bullet-like lines in experience/projects/skills/certifications/achievements separated by newline.",
                "Keep summary to 2-4 lines.",
            ],
        },
        "resume": resume_payload,
    }

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": json.dumps(prompt, ensure_ascii=False),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    try:
        request = urllib_request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = (
            payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        parsed = _extract_first_json_object(text)
        if not parsed:
            return resume_payload
        polished = normalize_resume_payload(parsed, resume_payload.get("full_name"))
        polished["email"] = resume_payload.get("email", "")
        polished["phone"] = resume_payload.get("phone", "")
        polished["location"] = resume_payload.get("location", "")
        polished["linkedin"] = resume_payload.get("linkedin", "")
        polished["github"] = resume_payload.get("github", "")
        return polished
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        return resume_payload


def build_resume_pdf_bytes(resume_payload: Dict[str, Any]):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PDF generation dependency is missing. Install reportlab to enable resume download.",
        )

    normalized = normalize_resume_payload(resume_payload)
    professional = polish_resume_with_gemini(normalized)

    def paragraph_text(value: str) -> str:
        return (value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    def to_lines(value: str) -> List[str]:
        raw = (value or "").replace("\r", "\n")
        split_lines = [line.strip(" -•\t") for line in raw.split("\n") if line.strip()]
        if len(split_lines) == 1:
            parts = [part.strip() for part in re.split(r"[;|]", split_lines[0]) if part.strip()]
            if len(parts) > 1:
                return parts
        return split_lines

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=34,
        bottomMargin=34,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ResumeTitle",
        parent=styles["Heading1"],
        fontSize=21,
        leading=24,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )
    section_style = ParagraphStyle(
        "ResumeSection",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=14,
        spaceAfter=3,
        spaceBefore=8,
        textColor=colors.HexColor("#115e59"),
    )
    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor("#111827"),
    )
    bullet_style = ParagraphStyle(
        "ResumeBullet",
        parent=body_style,
        leftIndent=10,
        bulletIndent=2,
        spaceBefore=1,
        spaceAfter=1,
    )

    story = []

    contact_parts = [
        professional.get("phone", ""),
        professional.get("email", ""),
        professional.get("linkedin", ""),
        professional.get("leetcode", ""),
        professional.get("github", ""),
    ]
    contact_lines = [part for part in contact_parts if part]

    header_left = [
        Paragraph(professional["full_name"] or "Student Resume", title_style),
    ]
    if professional.get("location"):
        header_left.append(Paragraph(paragraph_text(professional["location"]), body_style))
    if professional.get("headline"):
        header_left.append(Paragraph(paragraph_text(professional["headline"]), body_style))

    header_right = []
    for item in contact_lines:
        header_right.append(Paragraph(paragraph_text(item), body_style))

    if header_right:
        header_table = Table(
            [[header_left, header_right]],
            colWidths=[A4[0] * 0.62 - 42, A4[0] * 0.38 - 42],
            hAlign="LEFT",
        )
        header_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(header_table)
    else:
        story.append(Paragraph(professional["full_name"] or "Student Resume", title_style))
        if professional.get("headline"):
            story.append(Paragraph(paragraph_text(professional["headline"]), body_style))

    divider = Table([[""]], colWidths=[A4[0] - 84], rowHeights=[1])
    divider.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, -1), 0.6, colors.HexColor("#94a3b8")),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(divider)
    story.append(Spacer(1, 8))

    if professional.get("summary"):
        story.append(Paragraph("Profile Summary", section_style))
        story.append(Paragraph(paragraph_text(professional["summary"]), body_style))
        story.append(Spacer(1, 6))

    education_lines = [
        professional.get("education_1", ""),
        professional.get("education_2", ""),
        professional.get("education_3", ""),
    ]
    if not any(education_lines) and professional.get("education"):
        education_lines = to_lines(professional.get("education", ""))
    education_lines = [line for line in education_lines if line]
    if education_lines:
        story.append(Paragraph("Education", section_style))
        for line in education_lines:
            story.append(Paragraph(paragraph_text(line), body_style))
        story.append(Spacer(1, 6))

    skill_rows = [
        ("Programming", professional.get("skills_programming", "")),
        ("Frontend", professional.get("skills_frontend", "")),
        ("Backend", professional.get("skills_backend", "")),
        ("Databases", professional.get("skills_databases", "")),
        ("Tools", professional.get("skills_tools", "")),
    ]
    if any(value for _, value in skill_rows) or professional.get("skills"):
        story.append(Paragraph("Technical Skills", section_style))
        for label, value in skill_rows:
            if not value:
                continue
            story.append(Paragraph(paragraph_text(f"{label}: {value}"), body_style))
        if professional.get("skills"):
            story.append(Paragraph(paragraph_text(professional["skills"]), body_style))
        story.append(Spacer(1, 6))

    experience_title = professional.get("experience_1_title", "")
    experience_meta = professional.get("experience_1_meta", "")
    experience_points = professional.get("experience_1_points", "")
    if experience_title or experience_meta or experience_points or professional.get("experience"):
        story.append(Paragraph("Experience", section_style))
        if experience_title:
            story.append(Paragraph(paragraph_text(experience_title), body_style))
        if experience_meta:
            story.append(Paragraph(paragraph_text(experience_meta), body_style))
        exp_lines = to_lines(experience_points or professional.get("experience", ""))
        for line in exp_lines:
            story.append(Paragraph(paragraph_text(line), bullet_style, bulletText="•"))
        story.append(Spacer(1, 6))

    project_blocks = [
        (
            professional.get("project_1_title", ""),
            professional.get("project_1_link", ""),
            professional.get("project_1_points", ""),
        ),
        (
            professional.get("project_2_title", ""),
            professional.get("project_2_link", ""),
            professional.get("project_2_points", ""),
        ),
    ]
    has_structured_projects = any(title or link or points for title, link, points in project_blocks)
    if has_structured_projects or professional.get("projects"):
        story.append(Paragraph("Projects", section_style))
        if has_structured_projects:
            for title, link, points in project_blocks:
                if not (title or link or points):
                    continue
                heading = title or "Project"
                if link:
                    heading = f"{heading} ({link})"
                story.append(Paragraph(paragraph_text(heading), body_style))
                for line in to_lines(points):
                    story.append(Paragraph(paragraph_text(line), bullet_style, bulletText="•"))
                story.append(Spacer(1, 3))
        else:
            for line in to_lines(professional.get("projects", "")):
                story.append(Paragraph(paragraph_text(line), bullet_style, bulletText="•"))
        story.append(Spacer(1, 5))

    if professional.get("certifications"):
        story.append(Paragraph("Certifications", section_style))
        for line in to_lines(professional["certifications"]):
            story.append(Paragraph(paragraph_text(line), bullet_style, bulletText="•"))
        story.append(Spacer(1, 5))

    if professional.get("achievements"):
        story.append(Paragraph("Achievements", section_style))
        for line in to_lines(professional["achievements"]):
            story.append(Paragraph(paragraph_text(line), bullet_style, bulletText="•"))
        story.append(Spacer(1, 4))

    if len(story) <= 2:
        story.append(Paragraph("Add your resume details and download again.", body_style))

    document.build(story)
    return buffer.getvalue()


class PlacementInput(BaseModel):
    CGPA: float = Field(ge=0, le=10)
    Internships: float = Field(ge=0)
    Projects: float = Field(ge=0)
    Certifications: float = Field(ge=0)
    Communication_Skills: float = Field(ge=0, le=100)
    Aptitude_Score: float = Field(ge=0, le=100)
    Backlogs: float = Field(ge=0)


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateStudentRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    student_id: Optional[int] = None
    role: str = "student"
    admin_pin: Optional[str] = None

class AddStudentDataRequest(BaseModel):
    username: str
    password: str
    display_name: str
    student_id: int
    branch: str
    college_tier: str
    CGPA: float
    Internships: float
    Projects: float
    Certifications: float
    Communication_Skills: float
    Aptitude_Score: float
    Backlogs: float


BULK_STUDENT_FIELDS = [
    "username",
    "password",
    "display_name",
    "student_id",
    "branch",
    "college_tier",
    "CGPA",
    "Internships",
    "Projects",
    "Certifications",
    "Communication_Skills",
    "Aptitude_Score",
    "Backlogs",
]


class StudentProfileUpdateRequest(BaseModel):
    branch: Optional[str] = None
    college_tier: Optional[str] = None
    CGPA: float = Field(ge=0, le=10)
    Internships: float = Field(ge=0)
    Projects: float = Field(ge=0)
    Certifications: float = Field(ge=0)
    Communication_Skills: float = Field(ge=0, le=100)
    Aptitude_Score: float = Field(ge=0, le=100)
    Backlogs: float = Field(ge=0)


class JobPostingRequest(BaseModel):
    form_name: str
    job_link: str
    min_cgpa: float = Field(ge=0, le=10)
    max_backlogs: float = Field(ge=0)


class ResumeRequest(BaseModel):
    full_name: str = ""
    headline: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    leetcode: str = ""
    github: str = ""
    summary: str = ""
    education: str = ""
    education_1: str = ""
    education_2: str = ""
    education_3: str = ""
    experience: str = ""
    experience_1_title: str = ""
    experience_1_meta: str = ""
    experience_1_points: str = ""
    projects: str = ""
    project_1_title: str = ""
    project_1_link: str = ""
    project_1_points: str = ""
    project_2_title: str = ""
    project_2_link: str = ""
    project_2_points: str = ""
    skills: str = ""
    skills_programming: str = ""
    skills_frontend: str = ""
    skills_backend: str = ""
    skills_databases: str = ""
    skills_tools: str = ""
    certifications: str = ""
    achievements: str = ""


ROWS = load_rows()
STUDENT_INDEX = build_student_index(ROWS)


try:
    model = load_model()
    model_runtime = "saved-model"
    # attempt to reconstruct training feature columns by dummy-encoding the bundled dataset
    try:
        _df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
        # drop obvious identifiers and target columns if present
        drop_cols = [c for c in ['student_id', 'StudentID', 'Placement', 'salary_package_lpa', 'package_lpa', 'company_type'] if c in _df.columns]
        _df_X = _df.drop(columns=drop_cols, errors='ignore')
        # fillna with empty string for categorical consistency
        _df_X = _df_X.fillna("")
        _dummy = pd.get_dummies(_df_X)
        n_expected = getattr(model, 'n_features_in_', _dummy.shape[1])
        model_feature_columns = list(_dummy.columns[:n_expected])
        # build defaults for missing columns from modes / zeros
        feature_defaults = {}
        for col in _df_X.columns:
            if _df_X[col].dtype == object:
                try:
                    feature_defaults[col] = _df_X[col].mode()[0]
                except Exception:
                    feature_defaults[col] = ""
            else:
                try:
                    feature_defaults[col] = float(_df_X[col].mean())
                except Exception:
                    feature_defaults[col] = 0.0
    except Exception:
        model_feature_columns = None
        feature_defaults = {}
except Exception:
    model = FallbackModel()
    model_runtime = "fallback"
insights = build_insights(ROWS)
model_info = build_model_info(model)
model_info["runtime"] = model_runtime
if model_runtime == "fallback":
    model_info["notes"].append("Fallback scoring is active because the saved model could not be loaded in the current environment.")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "runtime": model_runtime}


@app.get("/api/insights")
def get_insights():
    return insights


@app.get("/api/project")
def get_project():
    return PROJECT_OVERVIEW


@app.get("/api/model-info")
def get_model_info():
    return model_info


@app.get("/api/site-data")
def get_site_data():
    return {
        "project": PROJECT_OVERVIEW,
        "model": model_info,
        "insights": insights,
    }


@app.post("/api/auth/register")
def register(payload: CreateStudentRequest):
    if payload.role == "admin":
        if payload.admin_pin != "1390":
            raise HTTPException(status_code=400, detail="Invalid admin PIN.")
        role = "admin"
        student_id = None
    else:
        role = "student"
        student_id = normalize_student_id(payload.student_id)
        if student_id is None:
            raise HTTPException(status_code=400, detail="Student ID is required for student registration.")
        if db_student_id_exists(student_id):
            raise HTTPException(status_code=400, detail="Student ID is already linked to another account.")
    
    # Actually register user in the db
    user = db_create_user(
        username=payload.username,
        password=payload.password,
        role=role,
        display_name=payload.display_name or payload.username,
        student_id=student_id,
    )
    if not user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Auto-login the user
    token = uuid4().hex
    session = get_user_public(user)
    session["token"] = token
    SESSIONS[token] = session

    profile = None
    if getattr(user, "role", None) == "student" and getattr(user, "student_id", None) is not None:
        student_row = get_row_for_student(user.student_id)
        if student_row:
            profile = build_student_profile(student_row)

    return {
        "message": "Registration successful",
        "token": token,
        "user": get_user_public(user),
        "profile": profile,
        "project": PROJECT_OVERVIEW,
        "model": model_info,
    }

@app.post("/api/auth/login")
def login(payload: LoginRequest):
    user = get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.get("role") == "student":
        sid = normalize_student_id(user.get("student_id"))
        if sid is not None and db_student_id_link_count(sid) > 1:
            raise HTTPException(
                status_code=409,
                detail="This student ID is linked to multiple accounts. Ask admin to assign unique student IDs.",
            )

    token = uuid4().hex
    session = get_user_public(user)
    session["token"] = token
    SESSIONS[token] = session

    profile = None
    if getattr(user, "role", None) == "student" and getattr(user, "student_id", None) is not None:
        student_row = get_row_for_student(user.student_id)
        if student_row:
            profile = build_student_profile(student_row)

    return {
        "token": token,
        "user": get_user_public(user),
        "profile": profile,
        "project": PROJECT_OVERVIEW,
        "model": model_info,
    }


@app.post("/api/auth/logout")
def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    token = current_user.get("token")
    if token and token in SESSIONS:
        SESSIONS.pop(token, None)
    return {"status": "ok"}


@app.get("/api/auth/me")
def me(current_user: Dict[str, Any] = Depends(get_current_user)):
    profile = None
    if current_user.get("role") == "student":
        student_row = get_row_for_student(current_user.get("student_id"))
        if student_row:
            profile = build_student_profile(student_row)
    return {
        "user": {
            "username": current_user["username"],
            "role": current_user["role"],
            "display_name": current_user.get("display_name") or current_user["username"],
            "student_id": current_user.get("student_id"),
        },
        "profile": profile,
    }


@app.get("/api/bootstrap")
def bootstrap(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    if not current_user:
        return {
            "user": None,
            "project": PROJECT_OVERVIEW,
            "model": model_info,
            "insights": insights,
            "students": [],
            "job_postings": db_list_job_postings(),
            "authenticated": False,
        }

    student_profiles = []
    if current_user.get("role") == "admin":
        users = db_list_users()
        for user in users:
            if user.get("role") == "student" and user.get("student_id") is not None:
                row = get_row_for_student(user.get("student_id"))
                if row:
                    payload = build_student_profile(row)
                    payload["username"] = user.get("username")
                    payload["display_name"] = user.get("display_name") or user.get("username")
                    student_profiles.append(payload)
    else:
        student_row = get_row_for_student(current_user.get("student_id"))
        if student_row:
            student_profiles = [build_student_profile(student_row)]
        else:
            student_profiles = [
                {
                    "student_id": normalize_student_id(current_user.get("student_id")),
                    "username": current_user.get("username"),
                    "display_name": current_user.get("display_name") or current_user.get("username"),
                    "branch": None,
                    "college_tier": None,
                    "placement": "Profile not created yet",
                    "probability": None,
                    "eligible_jobs": [],
                    "eligible_job_count": 0,
                    "readiness_score": None,
                    "profile": {},
                    "profile_missing": True,
                }
            ]

    return {
        "user": {
            "username": current_user["username"],
            "role": current_user["role"],
            "display_name": current_user.get("display_name") or current_user["username"],
            "student_id": current_user.get("student_id"),
        },
        "project": PROJECT_OVERVIEW,
        "model": model_info,
        "insights": insights,
        "students": student_profiles,
        "job_postings": db_list_job_postings(),
        "authenticated": True,
    }


@app.get("/api/students")
def list_students(current_user: Dict[str, Any] = Depends(require_admin)):
    student_profiles = []
    users = db_list_users()
    for user in users:
        if user.get("role") == "student" and user.get("student_id") is not None:
            row = get_row_for_student(user.get("student_id"))
            if row:
                payload = build_student_profile(row)
                payload["username"] = user.get("username")
                payload["display_name"] = user.get("display_name") or user.get("username")
                student_profiles.append(payload)
    return student_profiles


@app.get("/api/admin/departments")
def list_admin_departments(current_user: Dict[str, Any] = Depends(require_admin)):
    return {"departments": available_departments()}


@app.get("/api/admin/analytics")
def admin_analytics(
    department: Optional[str] = None,
    tier: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    normalized_department = (department or "").strip()
    normalized_tier = (tier or "").strip()
    normalized_start = (start_date or "").strip()
    normalized_end = (end_date or "").strip()

    filtered_rows, date_supported = filtered_analytics_rows(
        ROWS,
        department=normalized_department or None,
        tier=normalized_tier or None,
        start_date=normalized_start or None,
        end_date=normalized_end or None,
    )

    return {
        "filters": {
            "departments": available_departments(),
            "tiers": available_tiers(),
            "date_filter_supported": date_supported,
            "applied": {
                "department": normalized_department or None,
                "tier": normalized_tier or None,
                "start_date": normalized_start or None,
                "end_date": normalized_end or None,
            },
        },
        "insights": build_insights(filtered_rows, branch_min_students=1),
        "row_count": len(filtered_rows),
    }


@app.get("/api/admin/student-rankings")
def list_student_rankings(
    department: str,
    page: int = 1,
    page_size: int = 10,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    department = department.strip()
    if not department:
        raise HTTPException(status_code=400, detail="Department is required")

    page = max(1, int(page))
    page_size = min(50, max(1, int(page_size)))
    rankings = ranked_student_rows(department)
    total = len(rankings)
    page_count = max(1, (total + page_size - 1) // page_size)
    page = min(page, page_count)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "department": department,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "total": total,
        "students": rankings[start:end],
    }


@app.post("/api/admin/students")
def create_student(payload: AddStudentDataRequest, current_user: Dict[str, Any] = Depends(require_admin)):
    global insights
    exists = db_get_user_by_username(payload.username)
    if exists:
        raise HTTPException(status_code=400, detail="Username already exists")
    if db_student_id_exists(payload.student_id):
        raise HTTPException(status_code=400, detail="Student ID is already linked to another account")
    if get_row_for_student(payload.student_id):
        raise HTTPException(status_code=400, detail="Student ID already exists in dataset")
    
    new_row = student_row_from_values(payload.dict())
    
    add_student_row(new_row)
    persist_rows_to_dataset()
    
    # Recalculate statistics dynamically
    insights = build_insights(ROWS)
    
    # 2. Create actual User Authentication
    created = db_create_user(
        username=payload.username,
        password=payload.password,
        role="student",
        display_name=payload.display_name,
        student_id=payload.student_id,
    )
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create user account")
    return get_user_public(created)


@app.post("/api/admin/students/bulk-upload")
async def bulk_upload_students(file: UploadFile = File(...), current_user: Dict[str, Any] = Depends(require_admin)):
    global insights

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be encoded as UTF-8.")

    text = normalize_bulk_csv_text(text)
    reader = DictReader(StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV header row is required.")

    missing_headers = [field for field in BULK_STUDENT_FIELDS if field not in reader.fieldnames]
    if missing_headers:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {', '.join(missing_headers)}",
        )

    parsed_rows = []
    seen_student_ids = {}
    seen_usernames = {}
    for line_number, row in enumerate(reader, start=2):
        if not any(str(value or "").strip() for key, value in row.items() if key is not None):
            continue
        try:
            parsed = parse_bulk_student_row(row, line_number)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        student_id = parsed["student_id"]
        if student_id in seen_student_ids:
            first_line = seen_student_ids[student_id]
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate student ID {student_id} found in upload at line {line_number}; first seen at line {first_line}.",
            )
        seen_student_ids[student_id] = line_number

        username_key = parsed["username"].lower()
        if username_key in seen_usernames:
            first_line = seen_usernames[username_key]
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate username '{parsed['username']}' found in upload at line {line_number}; first seen at line {first_line}.",
            )
        seen_usernames[username_key] = line_number

        if db_student_id_exists(student_id):
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate student ID {student_id} already exists; found while processing line {line_number}.",
            )
        if db_get_user_by_username(parsed["username"]):
            raise HTTPException(
                status_code=400,
                detail=f"Username '{parsed['username']}' already exists; found while processing line {line_number}.",
            )

        parsed_rows.append(parsed)

    if not parsed_rows:
        raise HTTPException(status_code=400, detail="CSV does not contain any student rows.")

    created_users = []
    new_rows = []
    for parsed in parsed_rows:
        created = db_create_user(
            username=parsed["username"],
            password=parsed["password"],
            role="student",
            display_name=parsed["display_name"],
            student_id=parsed["student_id"],
        )
        if not created:
            raise HTTPException(status_code=500, detail=f"Failed to create account for {parsed['username']}.")
        created_users.append(created)
        new_rows.append(student_row_from_values(parsed))

    for row in new_rows:
        add_student_row(row)

    persist_rows_to_dataset()
    insights = build_insights(ROWS)

    return {
        "message": f"Uploaded {len(new_rows)} students successfully.",
        "created": len(created_users),
        "students": [get_user_public(user) for user in created_users],
    }


@app.post("/api/admin/job-postings")
def create_job_posting(payload: JobPostingRequest, current_user: Dict[str, Any] = Depends(require_admin)):
    form_name = payload.form_name.strip()
    job_link = payload.job_link.strip()
    if not form_name:
        raise HTTPException(status_code=400, detail="Form name is required")
    if not job_link:
        raise HTTPException(status_code=400, detail="Job link is required")

    posting = db_create_job_posting(
        form_name=form_name,
        job_link=job_link,
        min_cgpa=float(payload.min_cgpa),
        max_backlogs=float(payload.max_backlogs),
        created_by=current_user.get("username"),
    )
    if not posting:
        raise HTTPException(status_code=500, detail="Failed to create job posting")

    return posting


@app.put("/api/student/profile")
def update_own_student_profile(payload: StudentProfileUpdateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "student":
        raise HTTPException(status_code=403, detail="Student access required")

    student_id = normalize_student_id(current_user.get("student_id"))
    if student_id is None:
        raise HTTPException(status_code=400, detail="Student ID is missing")

    updated_profile = update_student_record(
        student_id,
        {
            "CGPA": float(payload.CGPA),
            "Internships": float(payload.Internships),
            "Projects": float(payload.Projects),
            "Certifications": float(payload.Certifications),
            "Communication_Skills": float(payload.Communication_Skills),
            "Aptitude_Score": float(payload.Aptitude_Score),
            "Backlogs": float(payload.Backlogs),
        },
    )

    return {
        "message": "Profile updated and prediction refreshed",
        "profile": updated_profile,
        "user": get_user_public(current_user),
    }


@app.get("/api/student/resume")
def get_student_resume(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    student_id = normalize_student_id(current_user.get("student_id"))
    if student_id is None:
        raise HTTPException(status_code=400, detail="Student ID is missing")

    resume_payload = db_get_resume(student_id)
    if resume_payload is None:
        resume_payload = default_resume_payload(current_user.get("display_name") or current_user.get("username"))
    else:
        resume_payload = normalize_resume_payload(
            resume_payload,
            current_user.get("display_name") or current_user.get("username"),
        )
    return {"resume": resume_payload}


@app.put("/api/student/resume")
def save_student_resume(payload: ResumeRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    student_id = normalize_student_id(current_user.get("student_id"))
    if student_id is None:
        raise HTTPException(status_code=400, detail="Student ID is missing")

    normalized = normalize_resume_payload(
        payload.dict(),
        current_user.get("display_name") or current_user.get("username"),
    )
    db_upsert_resume(student_id, normalized)
    return {"message": "Resume saved successfully.", "resume": normalized}


@app.get("/api/download/resume.pdf")
def download_resume_pdf(student_id: Optional[int] = None, current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") == "admin":
        target_id = normalize_student_id(student_id)
        if target_id is None:
            raise HTTPException(status_code=400, detail="student_id is required for admin download.")
        target_user = db_get_user_by_student_id(target_id)
    else:
        target_id = normalize_student_id(current_user.get("student_id"))
        if target_id is None:
            raise HTTPException(status_code=400, detail="Student ID is missing")
        target_user = db_get_user_by_student_id(target_id)

    resume_payload = db_get_resume(target_id)
    if resume_payload is None:
        raise HTTPException(status_code=404, detail="Resume not found. Save resume data first.")

    display_name = None
    if target_user:
        display_name = target_user.get("display_name") or target_user.get("username")
    normalized_resume = normalize_resume_payload(resume_payload, display_name)
    pdf_bytes = build_resume_pdf_bytes(normalized_resume)
    safe_name = "".join(character if character.isalnum() else "_" for character in (normalized_resume.get("full_name") or "resume")).strip("_") or "resume"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_resume.pdf"'},
    )


@app.get("/api/students/{student_id}")
def get_student(student_id: int, current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin" and normalize_student_id(current_user.get("student_id")) != student_id:
        raise HTTPException(status_code=403, detail="You can only access your own profile")
    row = get_row_for_student(student_id)
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return build_student_profile(row)


@app.get("/api/download/overall.csv")
def download_overall_csv(current_user: Dict[str, Any] = Depends(require_admin)):
    return csv_response("overall_statistics.csv", overall_stats_csv())


@app.get("/api/download/student.csv")
def download_student_csv(student_id: Optional[int] = None, current_user: Dict[str, Any] = Depends(get_current_user)):
    target_id = student_id if student_id is not None else normalize_student_id(current_user.get("student_id"))
    if current_user.get("role") != "admin" and target_id != normalize_student_id(current_user.get("student_id")):
        raise HTTPException(status_code=403, detail="You can only download your own record")
    row = get_row_for_student(target_id)
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return csv_response(f"student_{target_id}.csv", student_csv(row))


@app.get("/api/download/dataset.csv")
def download_dataset_csv(current_user: Dict[str, Any] = Depends(require_admin)):
    return FileResponse(DATASET_PATH, filename="placement_dataset.csv", media_type="text/csv")


@app.get("/api/download/predictions.csv")
def download_predictions_csv(current_user: Dict[str, Any] = Depends(require_admin)):
    return csv_response("dataset_predictions.csv", dataset_predictions_csv())


@app.get("/api/download/student-rankings.csv")
def download_student_rankings_csv(department: str, current_user: Dict[str, Any] = Depends(require_admin)):
    department = department.strip()
    if not department:
        raise HTTPException(status_code=400, detail="Department is required")
    safe_department = "".join(character if character.isalnum() else "_" for character in department).strip("_") or "department"
    return csv_response(f"student_rankings_{safe_department}.csv", student_rankings_csv(department))


@app.get("/api/download/improvement-report.csv")
def download_improvement_report_csv(
    scope: str = "overall",
    department: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    safe_scope = "".join(character if character.isalnum() else "_" for character in scope).strip("_") or "overall"
    safe_department = "".join(character if character.isalnum() else "_" for character in (department or "")).strip("_")
    filename = f"improvement_report_{safe_scope}"
    if safe_department:
        filename = f"{filename}_{safe_department}"
    return csv_response(f"{filename}.csv", improvement_report_csv(scope=scope, department=department))


@app.get("/api/download/academic-report.csv")
def download_academic_report_csv(
    scope: str = "overall",
    department: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    safe_scope = "".join(character if character.isalnum() else "_" for character in scope).strip("_") or "overall"
    safe_department = "".join(character if character.isalnum() else "_" for character in (department or "")).strip("_")
    filename = f"academic_report_{safe_scope}"
    if safe_department:
        filename = f"{filename}_{safe_department}"
    return csv_response(f"{filename}.csv", academic_report_csv(scope=scope, department=department))


@app.post("/api/predict")
def predict(data: PlacementInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    vector = core_vector_from_values(
        data.CGPA,
        data.Internships,
        data.Projects,
        data.Certifications,
        data.Communication_Skills,
        data.Aptitude_Score,
        data.Backlogs,
    )
    result_label, probability = model_predict_from_vector(vector)

    return {
        "result": result_label,
        "probability": round(probability * 100, 2) if probability is not None else None,
        "readiness_score": readiness_score(data),
        "recommendation": role_recommendation(data),
        "next_steps": [
            "Strengthen projects and internships",
            "Improve aptitude and communication practice",
            "Keep academic backlog count at zero",
        ],
    }
