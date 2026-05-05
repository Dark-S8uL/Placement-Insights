from collections import Counter, defaultdict
from csv import DictReader, DictWriter, reader as CsvReader
from io import StringIO
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
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


def build_insights(rows):
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
        if stats["total"] < 500:
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
