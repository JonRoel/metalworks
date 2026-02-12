from __future__ import annotations

import csv
import io
import os
import shutil
import sqlite3
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)

# -----------------------------
# Config (internal-only)
# -----------------------------
APP_HOST_DEV = "127.0.0.1"
APP_PORT = int(os.environ.get("METALWORKS_PORT", "5000"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get("METALWORKS_DB_PATH", os.path.join(BASE_DIR, "data", "inventory.db"))
BACKUP_DIR = os.environ.get("METALWORKS_BACKUP_DIR", os.path.join(BASE_DIR, "backups"))

print("DB_PATH:", DB_PATH)

#DB_PATH = os.environ.get("METALWORKS_DB_PATH", os.path.join("data", "inventory.db"))
#BACKUP_DIR = os.environ.get("METALWORKS_BACKUP_DIR", "backups")

ADMIN_PASSWORD = os.environ.get("METALWORKS_ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.environ.get("METALWORKS_SECRET_KEY", "change-this-to-a-long-random-string")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
MAX_ROWS = 25000

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# -----------------------------
# Used count for dropdown fields field management
# -----------------------------

LOOKUP_FIELD_USAGE = {
    "location": ("inventory", "location"),
    "supplier": ("inventory", "supplier"),
    "unit": ("inventory", "unit"),
    "grade": ("inventory", "grade"),
    "material": ("inventory", "description"),
    "description": ("inventory", "description"),
    # add more later, e.g.
    # "uom": ("inventory", "uom"),
}

# -----------------------------
# DB helpers
# -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            material TEXT NOT NULL,
            grade TEXT NOT NULL,
            description TEXT,
            location TEXT NOT NULL,
            length REAL NOT NULL CHECK(length >= 0),
            unit TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 1 CHECK(quantity >= 0),
            lbs_per_ft REAL CHECK(lbs_per_ft IS NULL OR lbs_per_ft >= 0),
            weight REAL CHECK(weight IS NULL OR weight >= 0),
            supplier TEXT,
            po_number TEXT,
            cost_lb REAL CHECK(cost_lb IS NULL OR cost_lb >= 0),
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_type TEXT NOT NULL,   -- 'adjust' or 'new'
            sku TEXT,
            material TEXT NOT NULL,
            grade TEXT NOT NULL,
            description TEXT,
            location TEXT NOT NULL,
            length REAL NOT NULL,
            unit TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 1 CHECK(quantity >= 0),
            lbs_per_ft REAL CHECK(lbs_per_ft IS NULL OR lbs_per_ft >= 0),
            weight REAL,
            supplier TEXT,
            po_number TEXT,
            cost_lb REAL CHECK(cost_lb IS NULL OR cost_lb >= 0),
            notes TEXT,
            old_data TEXT,   -- JSON snapshot at submission
            new_data TEXT,   -- JSON snapshot of requested values
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TEXT,
            reviewed_by TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT,
            action TEXT NOT NULL,         -- 'Adjustment', 'New SKU', 'Edited', etc.
            old_data TEXT,                -- JSON
            new_data TEXT,                -- JSON
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            actor TEXT                    -- 'shop' or 'admin'
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin','shop')),
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        """
    )

    cur.execute(
      """
      CREATE TABLE IF NOT EXISTS lookup_values (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          field_key TEXT NOT NULL,
          code TEXT NOT NULL,
          label TEXT,
          meta_num REAL DEFAULT 0,
          sort_order INTEGER DEFAULT 0,
          is_active INTEGER DEFAULT 1,
          UNIQUE(field_key, code)
      );
      """
    )

    # Add meta_json column if missing (SQLite has no IF NOT EXISTS for columns)
    # try:
    #     cur.execute("ALTER TABLE lookup_values ADD COLUMN meta_num REAL;")
    # except Exception:
    #     pass



    cur.execute("SELECT COUNT(*) AS c FROM locations;")
    if cur.fetchone()["c"] == 0:
        defaults = [
            ("SR1-A", "SR1-A"),
            ("SR1-B", "SR1-B"),
            ("SR1-C", "SR1-C"),
            ("SR2-D", "SR2-D"),
            ("FLOOR", "Shop Floor"),
        ]
        cur.executemany("INSERT INTO locations (code, label) VALUES (?, ?);", defaults)

    conn.commit()
    conn.close()

# Audit Log Query

def get_audit_logs(sku: str = "", limit: int = 200) -> list[dict]:
    limit = max(1, min(int(limit), 1000))

    conn = db()
    cur = conn.cursor()

    if sku:
        cur.execute(
            """
            SELECT * FROM audit_log
            WHERE sku = ?
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (sku, limit),
        )
    else:
        cur.execute(
            """
            SELECT * FROM audit_log
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,),
        )

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows



def backup_database(reason: str) -> Optional[str]:
    if os.path.exists(DB_PATH):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"inventory_{ts}_{reason}.db")
        shutil.copy2(DB_PATH, dst)
        return dst
    return None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

# -----------------------------
# Capture old vs new data for audit logging
# -----------------------------

import json

TRACK_FIELDS = ["material","grade","description","location","length","unit","quantity","lbs_per_ft","weight","supplier","po_number","cost_lb","notes"]

def snapshot_item_for_log(item: dict) -> dict:
    return {k: item.get(k) for k in (["sku"] + TRACK_FIELDS)}

TRACK_FIELDS = ["material","grade","description","location","length","unit","quantity","lbs_per_ft","weight","supplier","po_number","cost_lb","notes"]

def payload_to_new_snapshot(payload: dict) -> dict:
    out = {"sku": payload.get("sku")}
    for k in TRACK_FIELDS:
        v = payload.get(k)
        # Treat empty string as "not provided"
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def write_audit(sku: str, action: str, old_data: dict | None, new_data: dict | None, actor: str):
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit_log (sku, action, old_data, new_data, actor, created_at) VALUES (?, ?, ?, ?, ?, ?);",
        (
            sku,
            action,
            json.dumps(old_data) if old_data is not None else None,
            json.dumps(new_data) if new_data is not None else None,
            actor,
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()

# -----------------------------
# Auth (internal-only)
# -----------------------------
def is_admin() -> bool:
    return bool(session.get("is_admin") is True)


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_admin():
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)

    return wrapper

def redirect_admin_from_shop():
    if is_admin():
        return redirect(url_for("admin_inventory"))
    return None


# -----------------------------
# Inventory functions
# -----------------------------
def get_inventory() -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM inventory ORDER BY sku;")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_lookup(field_key: str, active_only=True) -> list[dict]:
    conn = db()
    cur = conn.cursor()

    if active_only:
        cur.execute(
            """
            SELECT code, COALESCE(label, code) AS label, meta_num
            FROM lookup_values
            WHERE field_key=? AND is_active=1
            ORDER BY sort_order, code;
            """,
            (field_key,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    cur.execute(
        """
        SELECT id, field_key, code, COALESCE(label, code) AS label, is_active, sort_order, meta_num
        FROM lookup_values
        WHERE field_key=?
        ORDER BY sort_order, code;
        """,
        (field_key,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    for r in rows:
        r["used_count"] = lookup_usage_count(r["field_key"], r["code"])
    return rows


def lookup_usage_count(field_key: str, code: str) -> int:
    mapping = LOOKUP_FIELD_USAGE.get(field_key)
    if not mapping:
        return 0
    table, col = mapping

    conn = db()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {col} = ?;", (code,))
    n = int(cur.fetchone()["c"])
    conn.close()
    return n


def get_item(item_id: int) -> Optional[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM inventory WHERE id=?;", (item_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def search_by_sku(q: str) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM inventory WHERE sku LIKE ? ORDER BY sku;", (f"%{q}%",))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_item(data: Dict[str, Any]) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inventory
        (sku, material, grade, description, length, unit, location, quantity, lbs_per_ft, weight, supplier, po_number, cost_lb, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            data["sku"],
            data["material"],
            data["grade"],
            data["description"],
            float(data["length"]),
            data["unit"],
            data["location"],
            float(data.get("quantity", 1) or 1),
            float(data["lbs_per_ft"]),
            float(data["weight"]) if str(data.get("weight", "")).strip() not in ("", "None", "none") else None,
            data.get("supplier") or None,
            data.get("po_number") or None,
            float(data["cost_lb"]),
            data.get("notes") or None,
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def update_item(item_id: int, data: Dict[str, Any]) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE inventory
        SET sku=?, material=?, grade=?, description=?, length=?, unit=?, location=?, quantity=?, lbs_per_ft=?, weight=?, supplier=?, po_number=?, cost_lb=?, notes=?, updated_at=?
        WHERE id=?;
        """,
        (
            data["sku"],
            data["material"],
            data["grade"],
            data["description"],
            float(data["length"]),
            data["unit"],
            data["location"],
            float(data.get("quantity", 1) or 1),
            float(data["lbs_per_ft"]),
            float(data["weight"]) if str(data.get("weight", "")).strip() not in ("", "None", "none") else None,
            data.get("supplier") or None,
            data.get("po_number") or None,
            float(data["cost_lb"]),
            data.get("notes") or None,
            now_iso(),
            item_id,
        ),
    )
    conn.commit()
    conn.close()


def delete_item(item_id: int) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM inventory WHERE id=?;", (item_id,))
    conn.commit()
    conn.close()


def next_sku() -> str:
    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT MAX(CAST(sku AS INTEGER)) AS m FROM inventory;")
    inv_row = cur.fetchone()
    inv_max = inv_row["m"] if inv_row and inv_row["m"] is not None else 0

    cur.execute("""
        SELECT MAX(CAST(sku AS INTEGER)) AS m
        FROM pending_requests
        WHERE request_type='new' AND status='pending';
    """)
    req_row = cur.fetchone()
    req_max = req_row["m"] if req_row and req_row["m"] is not None else 0

    conn.close()

    m = max(int(inv_max), int(req_max))
    return str(m + 1).zfill(4)

# -----------------------------
# Inventory Sort Functions
# -----------------------------

ALLOWED_SORT_COLS = {"sku", "material", "length", "location", "quantity", "weight", "supplier", "po_number"}

def query_inventory(filters: dict) -> list[dict]:
    """
    filters supports:
      include_zero: bool (default False)
      q: global search string
      sku/material/grade/description/location/supplier/po_number: partial match
      min_length/max_length, min_qty/max_qty (optional)
      sort: allowed column
      dir: asc|desc
      limit: int
    """
    include_zero = str(filters.get("include_zero", "")).lower() in ("1", "true", "yes", "on")
    q = (filters.get("q") or "").strip()

    sku = (filters.get("sku") or "").strip()
    grade = (filters.get("grade") or "").strip()
    material = (filters.get("material") or "").strip()
    location = (filters.get("location") or "").strip()
    supplier = (filters.get("supplier") or "").strip()
    po_number = (filters.get("po_number") or "").strip()

    sort = (filters.get("sort") or "sku").strip()
    direction = (filters.get("dir") or "asc").strip().lower()
    direction = "desc" if direction == "desc" else "asc"

    try:
        limit = int(filters.get("limit") or 5000)
    except Exception:
        limit = 5000
    limit = max(1, min(limit, 20000))

    where = []
    params = []

    if not include_zero:
        where.append("quantity > 0")

    # Global search across a couple columns, add more in the if: q to include additional fields. Don't forget to add a like.
    if q:
        where.append("(sku LIKE ? OR description LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    # Per-column filters
    if sku:
        where.append("sku LIKE ?")
        params.append(f"%{sku}%")
    if grade:
        where.append("grade LIKE ?")
        params.append(f"%{grade}%")
    if material:
        where.append("material LIKE ?")
        params.append(f"%{material}%")
    if location:
        where.append("location = ?")
        params.append(location)
    if supplier:
        where.append("supplier LIKE ?")
        params.append(f"%{supplier}%")
    if po_number:
        where.append("po_number LIKE ?")
        params.append(f"%{po_number}%")

    # Numeric optional filters
    def add_num_range(field: str, min_key: str, max_key: str):
        vmin = (filters.get(min_key) or "").strip()
        vmax = (filters.get(max_key) or "").strip()
        if vmin != "":
            where.append(f"{field} >= ?")
            params.append(float(vmin))
        if vmax != "":
            where.append(f"{field} <= ?")
            params.append(float(vmax))

    add_num_range("length", "min_length", "max_length")
    add_num_range("quantity", "min_qty", "max_qty")

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # Sort safety
    if sort not in ALLOWED_SORT_COLS:
        sort = "sku"

    sql = f"""
        SELECT * FROM inventory
        {where_sql}
        ORDER BY {sort} {direction}, sku ASC
        LIMIT ?;
    """
    params.append(limit)

    conn = db()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows



# -----------------------------
# Requests (shop -> admin workflow)
# -----------------------------
def submit_request(payload: Dict[str, Any]) -> int:
    # For adjustments: snapshot current inventory row as old_data
    old_data = None
    if payload["request_type"] == "adjust":
        sku = payload.get("sku", "").strip()
        if not sku:
            raise ValueError("SKU required for adjustment")

        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM inventory WHERE sku=?;", (sku,))
        item = cur.fetchone()
        conn.close()

        if not item:
            raise ValueError(f"SKU {sku} not found")

        old_data = snapshot_item_for_log(dict(item))

    new_data = payload_to_new_snapshot(payload)

    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO pending_requests
        (request_type, sku, material, grade, description, length, unit, location, quantity, lbs_per_ft, weight, supplier, po_number, cost_lb, notes, old_data, new_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            payload["request_type"],
            payload.get("sku"),
            payload["material"],
            payload["grade"],
            payload.get("description"),
            float(payload["length"]),
            payload["unit"],              
            payload["location"],          
            float(payload.get("quantity", 1) or 1),
            float(payload.get("lbs_per_ft", 0) or 0),
            float(payload["weight"]) if str(payload.get("weight", "")).strip() not in ("", "None", "none") else None,
            payload.get("supplier") or None,
            payload.get("po_number") or None,
            float(payload.get("cost_lb", 0) or 0),
            payload.get("notes") or None,
            json.dumps(old_data) if old_data else None,
            json.dumps(new_data),
        ),
)
    req_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(req_id)


def pending_requests() -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM pending_requests WHERE status='pending' ORDER BY created_at DESC;")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def approve_request(req_id: int, overrides: Optional[Dict[str, Any]] = None) -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM pending_requests WHERE id=?;", (req_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    req = dict(row)

    if req.get("status") != "pending":
        conn.close()
        raise ValueError(f"Request is already {req.get('status')}. Refresh the page.")

    old_data = json.loads(req["old_data"]) if req.get("old_data") else {}
    new_data = json.loads(req["new_data"]) if req.get("new_data") else {}

    # ✅ IMPORTANT: use the function argument, do NOT overwrite with request.get_json here
    overrides = overrides or {}

    # Apply overrides (SKU locked: ignore any sku in overrides)
    for k in TRACK_FIELDS:
        if k not in overrides:
            continue
        v = overrides[k]
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        new_data[k] = v

    def coerce_numbers(d: dict) -> None:
        if "length" in d and d["length"] not in (None, ""):
            d["length"] = float(d["length"])
        if "quantity" in d and d["quantity"] not in (None, ""):
            d["quantity"] = float(d["quantity"])
        if "lbs_per_ft" in d and d["lbs_per_ft"] not in (None, ""):
            d["lbs_per_ft"] = float(d["lbs_per_ft"])
        if "cost_lb" in d and d["cost_lb"] not in (None, ""):
            d["cost_lb"] = float(d["cost_lb"])
        if "weight" in d:
            w = d["weight"]
            d["weight"] = None if (w is None or (isinstance(w, str) and w.strip() == "")) else float(w)

    if req["request_type"] == "new":
        sku = req.get("sku")

        # Build a full merged view for NEW requests (row fields + new_data + overrides)
        merged = {**req, **old_data, **new_data}

        # Required by your inventory schema / add_item()
        new_item = {
            "sku": sku,
            "material": merged.get("material"),
            "grade": merged.get("grade"),
            "description": merged.get("description"),
            "length": merged.get("length"),
            "unit": merged.get("unit"),
            "location": merged.get("location"),
            "quantity": merged.get("quantity", 1),
            "lbs_per_ft": merged.get("lbs_per_ft", 0),
            "weight": merged.get("weight"),
            "supplier": merged.get("supplier"),
            "po_number": merged.get("po_number"),
            "cost_lb": merged.get("cost_lb", 0),
            "notes": merged.get("notes"),
        }

        # Validate minimums
        if not new_item["sku"] or not new_item["material"] or not new_item["grade"] or new_item["length"] in (None, "") or not new_item["unit"] or not new_item["location"]:
            conn.close()
            raise ValueError("New SKU request missing required fields (sku/material/grade/length/unit/location).")

        coerce_numbers(new_item)

        # Prevent duplicate SKU insert
        cur.execute("SELECT 1 FROM inventory WHERE sku=?;", (sku,))
        if cur.fetchone():
            conn.close()
            raise ValueError(f"SKU {sku} already exists in inventory. Decline this request and create a new one.")

        add_item(new_item)
        write_audit(sku, "New SKU", None, snapshot_item_for_log(new_item), actor="admin")

    elif req["request_type"] == "adjust":
        sku = req["sku"]
        cur.execute("SELECT * FROM inventory WHERE sku=?;", (sku,))
        inv = cur.fetchone()
        if inv:
            inv = dict(inv)
            true_old = snapshot_item_for_log(inv)

            # MERGE ONLY provided fields
            for k in TRACK_FIELDS:
                if k in new_data:
                    inv[k] = new_data[k]

            coerce_numbers(inv)
            update_item(int(inv["id"]), inv)

            write_audit(sku, "Adjustment", true_old, snapshot_item_for_log(inv), actor="admin")

    # Mark approved
    cur.execute(
        """
        UPDATE pending_requests
        SET status='approved', reviewed_at=?, reviewed_by=?
        WHERE id=?;
        """,
        (now_iso(), "admin", req_id),
    )
    conn.commit()
    conn.close()



def decline_request(req_id: int) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE pending_requests
        SET status='declined', reviewed_at=?, reviewed_by=?
        WHERE id=?;
        """,
        (now_iso(), "admin", req_id),
    )
    conn.commit()
    conn.close()


# -----------------------------
# CSV upload
# -----------------------------
def parse_inventory_csv(file_bytes: bytes) -> List[Dict[str, Any]]:
    text = file_bytes.decode("utf-8-sig").strip()
    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise ValueError("CSV missing headers")

    # Flexible header mapping
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"CSV exceeds max row limit ({MAX_ROWS})")

    items: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows, start=2):
        mapped: Dict[str, Any] = {}
        for k, v in r.items():
            if k is None:
                continue
            key = norm(k)
            val = (v or "").strip()
            if key in ("sku",):
                mapped["sku"] = val
            elif key in ("material", "description"):
                mapped["material"] = val
            elif key == "length":
                mapped["length"] = float(val) if val else 0
            elif key == "location":
                mapped["location"] = val
            elif key in ("qty", "quantity"):
                mapped["quantity"] = float(val) if val else 1
            elif key == "weight":
                mapped["weight"] = float(val) if val else None
            elif key in ("supplier", "vendor"):
                mapped["supplier"] = val
            elif key in ("po", "po_number"):
                mapped["po_number"] = val
            elif key in ("notes", "note"):
                mapped["notes"] = val

        # Required minimum
        if not mapped.get("sku") or not mapped.get("material"):
            continue
        if "length" not in mapped or "location" not in mapped:
            # These are required for our schema
            raise ValueError(f"Row {idx}: missing length or location")
        items.append(mapped)

    if not items:
        raise ValueError("No valid rows found (need sku + material + length + location)")
    return items


def bulk_upsert(items: List[Dict[str, Any]], replace_existing: bool) -> Dict[str, int]:
    conn = db()
    cur = conn.cursor()

    if replace_existing:
        cur.execute("DELETE FROM inventory;")

    imported = 0
    updated = 0

    for it in items:
        sku = it["sku"]
        cur.execute("SELECT id FROM inventory WHERE sku=?;", (sku,))
        existing = cur.fetchone()

        if existing:
            cur.execute(
                """
                UPDATE inventory
                SET material=?, grade=?, description=?, length=?, unit=?, lbs_per_ft=?, location=?, quantity=?, weight=?, supplier=?, po_number=?, cost_lb=?, notes=?, updated_at=?
                WHERE sku=?;
                """,
                (
                    it.get("material"),
                    it.get("grade"),
                    it.get("description"),
                    float(it.get("lbs_per_ft", 0) or 0),
                    float(it.get("length", 0) or 0),
                    it.get("location"),
                    float(it.get("quantity", 1) or 1),
                    float(it["weight"]) if str(it.get("weight", "")).strip() not in ("", "None", "none") else None,
                    it.get("supplier") or None,
                    it.get("po_number") or None,
                    it.get("cost_lb"),
                    it.get("notes") or None,
                    now_iso(),
                    sku,
                ),
            )
            updated += 1
        else:
            cur.execute(
                """
                INSERT INTO inventory
                (sku, material, grade, description, lbs_per_ft, length, unit, location, quantity, weight, supplier, po_number, cost_lb, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sku,
                    it.get("material"),
                    it.get("grade"),
                    it.get("description"),
                    float(it.get("length", 0) or 0),
                    it.get("unit"),
                    it.get("location"),
                    float(it.get("quantity", 1) or 1),
                    float(it.get("lbs_per_ft", 0) or 0),
                    float(it["weight"]) if str(it.get("weight", "")).strip() not in ("", "None", "none") else None,
                    it.get("supplier") or None,
                    it.get("po_number") or None,
                    float(it.get("cost_lb", 0) or 0),
                    it.get("notes") or None,
                    now_iso(),
                ),
            )
            imported += 1

    conn.commit()
    conn.close()
    return {"imported": imported, "updated": updated, "total": imported + updated}


# -----------------------------
# Pages
# -----------------------------
@app.get("/")
def home():
    if is_admin():
        return redirect(url_for("admin_inventory"))
    return redirect(url_for("shop_inventory"))


# Shop pages
@app.get("/shop/inventory")
def shop_inventory():
    r = redirect_admin_from_shop()
    if r: return r
    return render_template("shop_inventory.html", is_admin=is_admin(), active="shop_inventory")



@app.get("/shop/adjustment")
def shop_adjustment():
    r = redirect_admin_from_shop()
    if r: return r
    return render_template("shop_adjustment.html", is_admin=is_admin(), active="shop_adjustment")


@app.get("/shop/new-sku")
def shop_new_sku():
    r = redirect_admin_from_shop()
    if r: return r
    return render_template("shop_new_sku.html", is_admin=is_admin(), active="shop_new_sku")


# Admin pages
@app.get("/admin/login")
def admin_login():
    return render_template("admin_login.html", is_admin=is_admin(), active="admin_login")


@app.post("/admin/login")
def admin_login_post():
    data = request.get_json(silent=True) or {}
    if data.get("password") == ADMIN_PASSWORD:
        session["is_admin"] = True
        return jsonify(success=True)
    return jsonify(success=False, message="Invalid password"), 401


@app.get("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.get("/admin/inventory")
@admin_required
def admin_inventory():
    return render_template("admin_inventory.html", is_admin=True, active="admin_inventory")


@app.get("/admin/add")
@admin_required
def admin_add():
    return render_template("admin_add.html", is_admin=True, active="admin_add")


@app.get("/admin/upload")
@admin_required
def admin_upload():
    return render_template("admin_upload.html", is_admin=True, active="admin_upload")


@app.get("/admin/requests")
@admin_required
def admin_requests():
    return render_template("admin_requests.html", is_admin=True, active="admin_requests")


@app.get("/admin/edit/<int:item_id>")
@admin_required
def admin_edit(item_id: int):
    return render_template("admin_edit.html", is_admin=True, active="admin_inventory", item_id=item_id)


@app.get("/admin/audit")
@admin_required
def admin_audit():
    return render_template("admin_audit.html", is_admin=True, active="admin_audit")

@app.get("/admin/fields")
@admin_required
def admin_fields():
    return render_template("admin_fields.html", is_admin=True, active="admin_fields")

@app.get("/admin/reports")
@admin_required
def admin_reports():
    return render_template("admin_reports.html", is_admin=True, active="admin_reports")





# -----------------------------
# API
# -----------------------------

@app.get("/api/inventory")
def api_inventory():
    
    # Works for both shop and admin (admin sees same data, just different pages)
    return jsonify(query_inventory(request.args))


# @app.get("/api/inventory")
# def api_inventory():
#     q = (request.args.get("q") or "").strip()
#     if q:
#         return jsonify(search_by_sku(q))
#     return jsonify(get_inventory())


@app.get("/api/inventory/<int:item_id>")
def api_inventory_item(item_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM inventory WHERE id=?;", (item_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify(error="Not found"), 404
    return jsonify(dict(row))



@app.post("/api/inventory")
def api_inventory_add():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    data = request.get_json(silent=True) or {}
    add_item(data)
    return jsonify(success=True)


@app.put("/api/inventory/<int:item_id>")
def api_inventory_update(item_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    data = request.get_json(silent=True) or {}
    update_item(item_id, data)
    return jsonify(success=True)


@app.delete("/api/inventory/<int:item_id>")
def api_inventory_delete(item_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    backup_database("delete")
    delete_item(item_id)
    return jsonify(success=True)


@app.get("/api/next-sku")
def api_next_sku():
    return jsonify(sku=next_sku())


@app.post("/api/requests/submit")
def api_requests_submit():
    # Shop users submit requests (no admin required)
    data = request.get_json(silent=True) or {}

    req_type = (data.get("request_type") or "").strip().lower()
    if req_type not in ("adjust", "new"):
        return jsonify(success=False, error="Invalid request_type"), 400

    def has_value(k: str) -> bool:
        v = data.get(k)
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() != ""
        return True  # numbers/bools count

    # ----- ADJUST REQUESTS -----
    if req_type == "adjust":
        if not (data.get("sku") or "").strip():
            return jsonify(success=False, error="SKU required for adjustment"), 400

        # Require at least one change field
        # ✅ fixed comma bug and normalized cost field name to cost_lb
        change_fields = [
            "material", "grade", "description", "length", "unit", "location", "quantity",
            "lbs_per_ft", "weight", "supplier", "po_number", "cost_lb", "notes"
        ]

        if not any(has_value(f) for f in change_fields):
            return jsonify(success=False, error="Provide at least one field to change"), 400

        # unit is optional for adjust; if you want a default, uncomment:
        # data.setdefault("unit", "ft")

    # ----- NEW REQUESTS -----
    if req_type == "new":
        required = ["material", "grade", "length", "location", "unit"]
        missing = [k for k in required if not has_value(k)]
        if missing:
            return jsonify(success=False, error=f"Missing required fields: {', '.join(missing)}"), 400

        # Defaults
        if not has_value("quantity"):
            data["quantity"] = 1

    # Submit
    try:
        req_id = submit_request(data)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

    return jsonify(success=True, request_id=req_id)


# @app.post("/api/requests/submit")
# def api_requests_submit():
#     # Shop users submit requests (no admin required)
#     data = request.get_json(silent=True) or {}
#     # Basic validation
#     if data.get("request_type") not in ("adjust", "new"):
#         return jsonify(success=False, error="Invalid request_type"), 400
#     if data.get("request_type") == "adjust":
#       if not data.get("sku"):
#           return jsonify(success=False, error="SKU required for adjustment"), 400

#       # Require at least one change field to be provided
#       change_fields = ["material", "grade", "description", "length", "location", "quantity", "weight", "supplier", "po_number", "cost_lb", "notes"]
#       if not any(str(data.get(f, "")).strip() != "" for f in change_fields):
#           return jsonify(success=False, error="Provide at least one field to change"), 400

#     #if not data.get("material") or data.get("length") in (None, "") or not data.get("location"):
#     #    return jsonify(success=False, error="material, length, location required"), 400

#     req_id = submit_request(data)
#     return jsonify(success=True, request_id=req_id)
    



@app.get("/api/requests/pending")
def api_requests_pending():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    return jsonify(pending_requests())


@app.post("/api/requests/<int:req_id>/approve")
def api_requests_approve(req_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    overrides = request.get_json(silent=True) or {}
    try:
        backup_database("approve_request")
        approve_request(req_id, overrides=overrides if overrides else None)
        return jsonify(success=True)
    except ValueError as e:
        # Conflict / validation issues (like duplicate SKU)
        return jsonify(success=False, error=str(e)), 409
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.get("/api/audit")
def api_audit():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    sku = (request.args.get("sku") or "").strip()
    limit = request.args.get("limit") or "200"
    return jsonify(get_audit_logs(sku=sku, limit=int(limit)))


@app.post("/api/requests/<int:req_id>/decline")
def api_requests_decline(req_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    decline_request(req_id)
    return jsonify(success=True)


@app.post("/api/upload/inventory")
def api_upload_inventory():
    if not is_admin():
        return jsonify(success=False, error="Unauthorized"), 401

    if "file" not in request.files:
        return jsonify(success=False, error="No file uploaded"), 400

    replace_existing = request.form.get("replace_existing") == "true"
    f = request.files["file"]
    file_bytes = f.read()

    try:
        items = parse_inventory_csv(file_bytes)
        if replace_existing:
            backup_database("replace")
        result = bulk_upsert(items, replace_existing=replace_existing)
        return jsonify(success=True, **result)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400


@app.get("/api/export/csv")
def api_export_csv():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    items = get_inventory()
    out = io.StringIO()
    fields = ["sku", "material", "grade", "description", "unit", "length", "location", "quantity", "lbs_per_ft", "weight", "supplier", "po_number", "cost_lb", "notes"]
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    for it in items:
        w.writerow({k: it.get(k, "") for k in fields})

    mem = io.BytesIO(out.getvalue().encode("utf-8"))
    mem.seek(0)
    filename = f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)


@app.post("/api/backup")
def api_backup():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    path = backup_database("manual")
    return jsonify(success=True, path=path)

def get_locations(active_only: bool = True) -> list[dict]:
    conn = db()
    cur = conn.cursor()
    if active_only:
        cur.execute("SELECT code, COALESCE(label, code) AS label FROM locations WHERE is_active=1 ORDER BY code;")
    else:
        cur.execute("SELECT code, COALESCE(label, code) AS label, is_active FROM locations ORDER BY code;")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# Depicated maybe
# @app.get("/api/locations")
# def api_locations():
#     # both shop & admin can read
#     return jsonify(get_locations(active_only=True))
# end Depricated

@app.get("/api/admin/fields/<field_key>")
def api_admin_field_list(field_key: str):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401
    return jsonify(get_lookup(field_key, active_only=False))

@app.get("/api/lookups/<field_key>")
def api_lookups(field_key: str):
    return jsonify(get_lookup(field_key, active_only=True))

@app.post("/api/admin/fields/<field_key>")
def api_admin_field_add(field_key: str):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    label = (data.get("label") or "").strip() or None
    meta_num = data.get("meta_num")
    meta_num = float(meta_num) if meta_num not in (None, "",) else None
    sort_order = int(data.get("sort_order") or 0)

    if not code:
        return jsonify(success=False, error="code is required"), 400

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO lookup_values (field_key, code, label, sort_order, is_active, meta_num) VALUES (?, ?, ?, ?, 1, ?);",
        (field_key, code, label, sort_order, meta_num),
    )
    conn.commit()
    conn.close()
    return jsonify(success=True)


@app.put("/api/admin/field-value/<int:row_id>")
def api_admin_field_update(row_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip() or None
    sort_order = int(data.get("sort_order") or 0)
    meta_num = data.get("meta_num")
    meta_num = float(meta_num) if meta_num not in (None, "",) else None
    # meta_json = (data.get("meta_json") or "").strip() or None
    is_active = 1 if str(data.get("is_active", "1")).lower() in ("1","true","yes","on") else 0

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE lookup_values SET label=?, sort_order=?, is_active=?, meta_num=? WHERE id=?;",
        (label, sort_order, is_active, meta_num, row_id),
    )
    conn.commit()

    updated = cur.rowcount
    conn.close()

    if updated == 0:
        return jsonify(success=False, error="No rows updated (bad id?)"), 404

    return jsonify(success=True, updated=updated)
    # cur = conn.cursor()
    # cur.execute(
    #     "UPDATE lookup_values SET label=?, sort_order=?, is_active=?, meta_num=? WHERE id=?;",
    #     (label, sort_order, is_active, row_id, meta_num),
    # )
    # conn.commit()
    # conn.close()
    # return jsonify(success=True)


@app.delete("/api/admin/field-value/<int:row_id>")
def api_admin_field_delete(row_id: int):
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT field_key, code FROM lookup_values WHERE id=?;", (row_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify(success=False, error="Not found"), 404

    field_key = row["field_key"]
    code = row["code"]

    used = lookup_usage_count(field_key, code)
    if used > 0:
        conn.close()
        return jsonify(success=False, error=f"Cannot delete: value is in use ({used} inventory rows). Disable it instead."), 409

    cur.execute("DELETE FROM lookup_values WHERE id=?;", (row_id,))
    conn.commit()
    conn.close()
    return jsonify(success=True)


@app.get("/api/admin/field-keys")
def api_admin_field_keys():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT field_key FROM lookup_values ORDER BY field_key;")
    keys = [r["field_key"] for r in cur.fetchall()]
    conn.close()
    return jsonify(keys)

#  --- Reports ---

@app.get("/api/reports/summary")
def api_reports_summary():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT
        COUNT(*) AS total_rows,
        SUM(CASE WHEN COALESCE(quantity,0) > 0 THEN 1 ELSE 0 END) AS in_stock_rows,
        SUM(COALESCE(quantity,0)) AS total_qty,
        SUM(COALESCE(weight,0)) AS total_weight,
        SUM(COALESCE(weight,0) * COALESCE(cost_lb,0)) AS total_value
      FROM inventory;
    """)
    row = cur.fetchone()
    conn.close()
    return jsonify(dict(row))


@app.get("/api/reports/by_location")
def api_reports_by_location():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT
        location,
        COUNT(*) AS sku_count,
        SUM(COALESCE(quantity,0)) AS total_qty,
        SUM(COALESCE(weight,0)) AS total_weight,
        SUM(COALESCE(weight,0) * COALESCE(cost_lb,0)) AS total_value
      FROM inventory
      GROUP BY location
      ORDER BY total_weight DESC, location ASC;
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.get("/api/reports/by_material")
def api_reports_by_material():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT
        material,
        COUNT(*) AS sku_count,
        SUM(COALESCE(quantity,0)) AS total_qty,
        SUM(COALESCE(weight,0)) AS total_weight,
        SUM(COALESCE(weight,0) * COALESCE(cost_lb,0)) AS total_value
      FROM inventory
      GROUP BY material
      ORDER BY total_weight DESC, material ASC;
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.get("/api/reports/by_grade")
def api_reports_by_grade():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT
        grade,
        COUNT(*) AS sku_count,
        SUM(COALESCE(quantity,0)) AS total_qty,
        SUM(COALESCE(weight,0)) AS total_weight,
        SUM(COALESCE(weight,0) * COALESCE(cost_lb,0)) AS total_value
      FROM inventory
      GROUP BY grade
      ORDER BY total_weight DESC, grade ASC;
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.get("/api/reports/top_descriptions")
def api_reports_top_descriptions():
    if not is_admin():
        return jsonify(error="Unauthorized"), 401

    limit = request.args.get("limit", "20")
    try:
        limit_n = max(1, min(200, int(limit)))
    except ValueError:
        limit_n = 20

    conn = db()
    cur = conn.cursor()
    cur.execute(f"""
      SELECT
        description,
        COUNT(*) AS sku_count,
        SUM(COALESCE(quantity,0)) AS total_qty,
        SUM(COALESCE(weight,0)) AS total_weight,
        SUM(COALESCE(weight,0) * COALESCE(cost_lb,0)) AS total_value
      FROM inventory
      GROUP BY description
      ORDER BY total_weight DESC, description ASC
      LIMIT ?;
    """, (limit_n,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)


# -----------------------------
# Startup + Run
# -----------------------------
init_db()

if __name__ == "__main__":
    # Dev run (Mac): easy logs, auto-reload
    app.run(host=APP_HOST_DEV, port=APP_PORT, debug=True)
