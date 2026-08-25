#!/usr/bin/env python3
"""
NEXUS · Artemis-X Mission Control  (Enhanced Edition)
New features:
  - Telemetry analytics  (/api/telemetry/stats, /api/telemetry/range)
  - AI anomaly explanation  (auto Claude call on fault injection)
  - AI Memory / log search  (/api/logs/search  +  FTS5)
  - Mission Timeline replay  (/api/telemetry/range)
  - Crew health monitoring  (spO2, respRate, stressIndex)
  - Multi-mission support  (/api/missions  CRUD)
  - Auto AI Mission Briefing  (every 5 min via WebSocket)
  - Export CSV & AI report  (/api/export/csv  /api/export/report)
  - Telemetry retention policy  (keep last 7 days, prune on boot+daily)
  - Predictive threshold alerts  (rolling z-score anomaly detection)
  - Voice-ready alert endpoint  (/api/alerts/latest)
Run: python server.py
Then open: http://localhost:8080
"""

import os
import json
import random
import sqlite3
import threading
import time
import csv
import io
import math
import statistics
from datetime import datetime, timedelta
from collections import deque

import urllib.request
import urllib.error

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_socketio import SocketIO

# ── Config ─────────────────────────────────────────────────────────────────────
PORT                   = 8080
BASE_DIR               = os.path.dirname(os.path.abspath(__file__))
DB_PATH                = os.path.join(BASE_DIR, "nexus.db")
ANTHROPIC_API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
TELEMETRY_INTERVAL_SEC = 2.0
BRIEFING_INTERVAL_SEC  = 300        # auto AI briefing every 5 minutes
RETENTION_DAYS         = 7          # keep telemetry for 7 days
ROLLING_WINDOW         = 20         # samples used for z-score anomaly detection
ZSCORE_THRESHOLD       = 2.8        # flag if reading deviates > 2.8 std devs

app     = Flask(__name__, static_folder=BASE_DIR)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared live state ──────────────────────────────────────────────────────────
sim_lock  = threading.Lock()
sim_state = {
    "injected":        {"thermal": 0, "comm": 0, "power": 0},
    "active_mission":  1,           # default mission id
}

# Rolling buffers for z-score detection  {sensor: deque([float, ...])}
rolling_buffers: dict[str, deque] = {}
# Latest alerts (for voice endpoint)
latest_alerts: list[dict] = []

# ── Database setup ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        -- ── core tables ────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS missions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            description TEXT,
            status      TEXT    NOT NULL DEFAULT 'active',
            created     TEXT    NOT NULL,
            archived    TEXT
        );

        CREATE TABLE IF NOT EXISTS mission_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            timestamp   TEXT    NOT NULL,
            level       TEXT    NOT NULL DEFAULT 'INFO',
            message     TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mission_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            met         TEXT    NOT NULL,
            type        TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            created     TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            session     TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS telemetry_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            timestamp   TEXT    NOT NULL,
            data        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS anomaly_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            kind        TEXT    NOT NULL,
            severity    TEXT    NOT NULL DEFAULT 'WARN',
            start_ts    TEXT    NOT NULL,
            end_ts      TEXT,
            explanation TEXT
        );

        CREATE TABLE IF NOT EXISTS crew_health (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            timestamp   TEXT    NOT NULL,
            heart_rate  REAL,
            spo2        REAL,
            resp_rate   REAL,
            stress_idx  REAL
        );

        CREATE TABLE IF NOT EXISTS mission_briefings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id  INTEGER NOT NULL DEFAULT 1,
            timestamp   TEXT    NOT NULL,
            content     TEXT    NOT NULL
        );

        -- ── FTS5 full-text search over logs ────────────────────────────────
        CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts
            USING fts5(message, content=mission_logs, content_rowid=id);

        -- ── seed default mission if empty ───────────────────────────────────
        INSERT OR IGNORE INTO missions (id, name, description, status, created)
            VALUES (1, 'Artemis-X', 'Primary lunar mission', 'active',
                    datetime('now'));
    """)
    con.commit()
    con.close()
    print("  ✓  Database ready →", DB_PATH)


def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def active_mission() -> int:
    with sim_lock:
        return sim_state["active_mission"]


# ── Telemetry retention (prune rows older than RETENTION_DAYS) ─────────────────
def prune_old_telemetry():
    cutoff = (datetime.utcnow() - timedelta(days=RETENTION_DAYS)).isoformat()
    con = get_db()
    deleted = con.execute(
        "DELETE FROM telemetry_snapshots WHERE timestamp < ?", (cutoff,)
    ).rowcount
    con.commit(); con.close()
    if deleted:
        print(f"  ✓  Pruned {deleted} old telemetry rows (>{RETENTION_DAYS}d)")


def retention_loop():
    """Run retention prune once a day."""
    while True:
        time.sleep(86_400)
        prune_old_telemetry()


# ── Anthropic helper ───────────────────────────────────────────────────────────
def call_anthropic(payload: dict):
    if not ANTHROPIC_API_KEY:
        return None, "ANTHROPIC_API_KEY not set."

    body = json.dumps({
        "model":      payload.get("model", "claude-sonnet-4-6"),
        "max_tokens": payload.get("max_tokens", 1000),
        "system":     payload.get("system", ""),
        "messages":   payload.get("messages", []),
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, f"Anthropic API error {e.code}: {e.read().decode()}"
    except Exception as e:
        return None, str(e)


# ── Static routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "space_ai.html"))


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


# ── Claude AI proxy ────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True)
    result, err = call_anthropic(payload)
    if err:
        return jsonify({"error": err}), 500

    reply   = result.get("content", [{}])[0].get("text", "")
    session = payload.get("session", "analyst")
    mid     = active_mission()
    messages = payload.get("messages", [])
    if messages:
        last_user = messages[-1].get("content", "")
        con = get_db()
        con.execute(
            "INSERT INTO chat_history (mission_id, session, role, content, timestamp) VALUES (?,?,?,?,?)",
            (mid, session, "user", last_user, datetime.utcnow().isoformat()),
        )
        con.execute(
            "INSERT INTO chat_history (mission_id, session, role, content, timestamp) VALUES (?,?,?,?,?)",
            (mid, session, "assistant", reply, datetime.utcnow().isoformat()),
        )
        con.commit(); con.close()

    return jsonify(result)


@app.route("/api/planner", methods=["POST"])
def api_planner():
    payload = request.get_json(force=True)
    result, err = call_anthropic(payload)
    if err:
        return jsonify({"error": err}), 500

    # Save planner history too
    reply   = result.get("content", [{}])[0].get("text", "")
    mid     = active_mission()
    messages = payload.get("messages", [])
    if messages:
        last_user = messages[-1].get("content", "")
        con = get_db()
        con.execute(
            "INSERT INTO chat_history (mission_id, session, role, content, timestamp) VALUES (?,?,?,?,?)",
            (mid, "planner", "user", last_user, datetime.utcnow().isoformat()),
        )
        con.execute(
            "INSERT INTO chat_history (mission_id, session, role, content, timestamp) VALUES (?,?,?,?,?)",
            (mid, "planner", "assistant", reply, datetime.utcnow().isoformat()),
        )
        con.commit(); con.close()

    return jsonify(result)


# ── Missions CRUD ──────────────────────────────────────────────────────────────
@app.route("/api/missions", methods=["GET"])
def list_missions():
    con = get_db()
    rows = con.execute("SELECT * FROM missions ORDER BY id DESC").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/missions", methods=["POST"])
def create_mission():
    data = request.get_json(force=True)
    name = data.get("name", "Unnamed Mission")
    desc = data.get("description", "")
    ts   = datetime.utcnow().isoformat()
    con  = get_db()
    cur  = con.execute(
        "INSERT INTO missions (name, description, status, created) VALUES (?,?,?,?)",
        (name, desc, "active", ts),
    )
    mid = cur.lastrowid
    con.commit(); con.close()
    return jsonify({"status": "created", "mission_id": mid}), 201


@app.route("/api/missions/<int:mid>", methods=["GET"])
def get_mission(mid):
    con  = get_db()
    row  = con.execute("SELECT * FROM missions WHERE id=?", (mid,)).fetchone()
    con.close()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/missions/<int:mid>/activate", methods=["POST"])
def activate_mission(mid):
    con = get_db()
    row = con.execute("SELECT id FROM missions WHERE id=?", (mid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "not found"}), 404
    con.close()
    with sim_lock:
        sim_state["active_mission"] = mid
    socketio.emit("mission_changed", {"mission_id": mid})
    return jsonify({"status": "active", "mission_id": mid})


@app.route("/api/missions/<int:mid>/archive", methods=["POST"])
def archive_mission(mid):
    ts  = datetime.utcnow().isoformat()
    con = get_db()
    con.execute(
        "UPDATE missions SET status='archived', archived=? WHERE id=?", (ts, mid)
    )
    con.commit(); con.close()
    return jsonify({"status": "archived"})


# ── Mission Logs ───────────────────────────────────────────────────────────────
@app.route("/api/logs", methods=["GET"])
def get_logs():
    limit = request.args.get("limit", 100, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    con   = get_db()
    rows  = con.execute(
        "SELECT * FROM mission_logs WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/logs", methods=["POST"])
def add_log():
    data    = request.get_json(force=True)
    ts      = datetime.utcnow().isoformat()
    level   = data.get("level", "INFO")
    message = data.get("message", "")
    mid     = active_mission()
    con     = get_db()
    cur     = con.execute(
        "INSERT INTO mission_logs (mission_id, timestamp, level, message) VALUES (?,?,?,?)",
        (mid, ts, level, message),
    )
    log_id = cur.lastrowid
    # Update FTS index
    con.execute(
        "INSERT INTO logs_fts (rowid, message) VALUES (?,?)", (log_id, message)
    )
    con.commit(); con.close()
    socketio.emit("log", {"id": log_id, "timestamp": ts, "level": level, "message": message})
    return jsonify({"status": "ok"}), 201


@app.route("/api/logs", methods=["DELETE"])
def clear_logs():
    mid = active_mission()
    con = get_db()
    con.execute("DELETE FROM mission_logs WHERE mission_id=?", (mid,))
    con.execute("DELETE FROM logs_fts")   # rebuild on next insert
    con.commit(); con.close()
    return jsonify({"status": "cleared"})


# ── NEW: Full-text log search ──────────────────────────────────────────────────
@app.route("/api/logs/search", methods=["GET"])
def search_logs():
    """
    Search mission logs using SQLite FTS5.
    ?q=thermal&limit=20
    """
    q     = request.args.get("q", "").strip()
    limit = request.args.get("limit", 50, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    if not q:
        return jsonify({"error": "q is required"}), 400

    con  = get_db()
    rows = con.execute(
        """
        SELECT ml.* FROM mission_logs ml
        JOIN logs_fts fts ON fts.rowid = ml.id
        WHERE fts.message MATCH ? AND ml.mission_id = ?
        ORDER BY ml.id DESC LIMIT ?
        """,
        (q, mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


# ── Mission Events ─────────────────────────────────────────────────────────────
@app.route("/api/events", methods=["GET"])
def get_events():
    mid  = request.args.get("mission_id", active_mission(), type=int)
    con  = get_db()
    rows = con.execute(
        "SELECT * FROM mission_events WHERE mission_id=? ORDER BY id DESC", (mid,)
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/events", methods=["POST"])
def add_event():
    data    = request.get_json(force=True)
    met     = data.get("met", "")
    etype   = data.get("type", "INFO")
    title   = data.get("title", "")
    created = datetime.utcnow().isoformat()
    mid     = active_mission()
    con     = get_db()
    cur     = con.execute(
        "INSERT INTO mission_events (mission_id, met, type, title, created) VALUES (?,?,?,?,?)",
        (mid, met, etype, title, created),
    )
    event_id = cur.lastrowid
    con.commit(); con.close()
    socketio.emit("event", {"id": event_id, "met": met, "type": etype, "title": title, "created": created})
    return jsonify({"status": "ok"}), 201


@app.route("/api/events/<int:event_id>", methods=["DELETE"])
def delete_event(event_id):
    con = get_db()
    con.execute("DELETE FROM mission_events WHERE id=?", (event_id,))
    con.commit(); con.close()
    return jsonify({"status": "deleted"})


# ── Telemetry Snapshots ────────────────────────────────────────────────────────
@app.route("/api/telemetry", methods=["POST"])
def save_telemetry():
    data = request.get_json(force=True)
    mid  = active_mission()
    con  = get_db()
    con.execute(
        "INSERT INTO telemetry_snapshots (mission_id, timestamp, data) VALUES (?,?,?)",
        (mid, datetime.utcnow().isoformat(), json.dumps(data)),
    )
    con.commit(); con.close()
    return jsonify({"status": "saved"}), 201


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    limit = request.args.get("limit", 50, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    con   = get_db()
    rows  = con.execute(
        "SELECT * FROM telemetry_snapshots WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, limit),
    ).fetchall()
    con.close()
    return jsonify([
        {"id": r["id"], "timestamp": r["timestamp"], "data": json.loads(r["data"])}
        for r in rows
    ])


# ── NEW: Telemetry time-range (timeline replay) ────────────────────────────────
@app.route("/api/telemetry/range", methods=["GET"])
def telemetry_range():
    """
    ?from=2024-01-01T00:00:00&to=2024-01-01T01:00:00&mission_id=1
    Returns telemetry snapshots between two timestamps — enables timeline replay.
    """
    from_ts = request.args.get("from")
    to_ts   = request.args.get("to")
    mid     = request.args.get("mission_id", active_mission(), type=int)
    limit   = request.args.get("limit", 500, type=int)

    if not from_ts or not to_ts:
        return jsonify({"error": "from and to are required"}), 400

    con  = get_db()
    rows = con.execute(
        """SELECT * FROM telemetry_snapshots
           WHERE mission_id=? AND timestamp BETWEEN ? AND ?
           ORDER BY timestamp ASC LIMIT ?""",
        (mid, from_ts, to_ts, limit),
    ).fetchall()
    con.close()
    return jsonify([
        {"id": r["id"], "timestamp": r["timestamp"], "data": json.loads(r["data"])}
        for r in rows
    ])


# ── NEW: Telemetry statistics ──────────────────────────────────────────────────
@app.route("/api/telemetry/stats", methods=["GET"])
def telemetry_stats():
    """
    Returns min/max/avg/latest for each sensor over the last N snapshots.
    ?samples=100&mission_id=1
    """
    samples = request.args.get("samples", 100, type=int)
    mid     = request.args.get("mission_id", active_mission(), type=int)
    con     = get_db()
    rows    = con.execute(
        "SELECT data FROM telemetry_snapshots WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, samples),
    ).fetchall()
    con.close()

    if not rows:
        return jsonify({"error": "no data"}), 404

    # Collect per-sensor values
    buckets: dict[str, list[float]] = {}
    for r in rows:
        snap = json.loads(r["data"])
        for k, v in snap.items():
            if k == "timestamp":
                continue
            try:
                buckets.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass

    stats = {}
    for sensor, vals in buckets.items():
        stats[sensor] = {
            "min":    round(min(vals), 2),
            "max":    round(max(vals), 2),
            "avg":    round(statistics.mean(vals), 2),
            "stdev":  round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
            "latest": vals[0],          # rows are DESC so first = most recent
            "samples": len(vals),
        }

    return jsonify({"mission_id": mid, "stats": stats})


# ── NEW: Crew Health ───────────────────────────────────────────────────────────
@app.route("/api/crew/health", methods=["GET"])
def get_crew_health():
    limit = request.args.get("limit", 50, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    con   = get_db()
    rows  = con.execute(
        "SELECT * FROM crew_health WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/crew/health/latest", methods=["GET"])
def get_crew_health_latest():
    mid  = request.args.get("mission_id", active_mission(), type=int)
    con  = get_db()
    row  = con.execute(
        "SELECT * FROM crew_health WHERE mission_id=? ORDER BY id DESC LIMIT 1", (mid,)
    ).fetchone()
    con.close()
    if not row:
        return jsonify({"error": "no data"}), 404
    return jsonify(dict(row))


# ── NEW: Anomaly events log ────────────────────────────────────────────────────
@app.route("/api/anomalies", methods=["GET"])
def get_anomalies():
    limit = request.args.get("limit", 50, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    con   = get_db()
    rows  = con.execute(
        "SELECT * FROM anomaly_events WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


# ── NEW: Export CSV ────────────────────────────────────────────────────────────
@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Download all telemetry for the active mission as a CSV file."""
    mid  = request.args.get("mission_id", active_mission(), type=int)
    con  = get_db()
    rows = con.execute(
        "SELECT timestamp, data FROM telemetry_snapshots WHERE mission_id=? ORDER BY id ASC",
        (mid,),
    ).fetchall()
    con.close()

    if not rows:
        return jsonify({"error": "no telemetry to export"}), 404

    # Build CSV in memory
    output = io.StringIO()
    first_data = json.loads(rows[0]["data"])
    fieldnames = ["timestamp"] + [k for k in first_data if k != "timestamp"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for r in rows:
        snap = json.loads(r["data"])
        snap["timestamp"] = r["timestamp"]
        writer.writerow({f: snap.get(f, "") for f in fieldnames})

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=nexus_mission_{mid}_telemetry.csv"},
    )


# ── NEW: AI Mission Report ─────────────────────────────────────────────────────
@app.route("/api/export/report", methods=["GET"])
def export_report():
    """Ask Claude to write a full mission status report based on recent data."""
    mid = request.args.get("mission_id", active_mission(), type=int)

    # Gather context
    con = get_db()
    mission = con.execute("SELECT * FROM missions WHERE id=?", (mid,)).fetchone()
    logs    = con.execute(
        "SELECT level, message FROM mission_logs WHERE mission_id=? ORDER BY id DESC LIMIT 30",
        (mid,),
    ).fetchall()
    events  = con.execute(
        "SELECT met, type, title FROM mission_events WHERE mission_id=? ORDER BY id DESC LIMIT 10",
        (mid,),
    ).fetchall()
    anomalies = con.execute(
        "SELECT kind, severity, start_ts, explanation FROM anomaly_events WHERE mission_id=? ORDER BY id DESC LIMIT 10",
        (mid,),
    ).fetchall()
    telem = con.execute(
        "SELECT data FROM telemetry_snapshots WHERE mission_id=? ORDER BY id DESC LIMIT 1",
        (mid,),
    ).fetchone()
    con.close()

    ctx_logs     = "\n".join(f"[{r['level']}] {r['message']}" for r in logs)
    ctx_events   = "\n".join(f"{r['met']} | {r['type']} | {r['title']}" for r in events)
    ctx_anomalies= "\n".join(
        f"{r['kind']} ({r['severity']}) @ {r['start_ts']}: {r['explanation'] or 'no explanation'}"
        for r in anomalies
    )
    ctx_telem    = json.dumps(json.loads(telem["data"]), indent=2) if telem else "unavailable"

    system_prompt = (
        "You are NEXUS Mission Director AI. Write a concise but thorough mission status report "
        "for the flight director. Include: overall mission health, key anomalies, crew status, "
        "telemetry highlights, and recommended actions. Use a professional aerospace tone. "
        "Format with clear sections."
    )
    user_msg = (
        f"Mission: {dict(mission)['name'] if mission else 'Unknown'}\n\n"
        f"=== Recent Logs ===\n{ctx_logs or 'None'}\n\n"
        f"=== Mission Events ===\n{ctx_events or 'None'}\n\n"
        f"=== Anomaly History ===\n{ctx_anomalies or 'None'}\n\n"
        f"=== Latest Telemetry ===\n{ctx_telem}\n\n"
        "Write the mission status report now."
    )

    result, err = call_anthropic({
        "system":     system_prompt,
        "max_tokens": 1500,
        "messages":   [{"role": "user", "content": user_msg}],
    })
    if err:
        return jsonify({"error": err}), 500

    report = result.get("content", [{}])[0].get("text", "")
    return jsonify({"mission_id": mid, "report": report, "generated": datetime.utcnow().isoformat()})


# ── NEW: Mission Briefings ─────────────────────────────────────────────────────
@app.route("/api/briefings", methods=["GET"])
def get_briefings():
    limit = request.args.get("limit", 10, type=int)
    mid   = request.args.get("mission_id", active_mission(), type=int)
    con   = get_db()
    rows  = con.execute(
        "SELECT * FROM mission_briefings WHERE mission_id=? ORDER BY id DESC LIMIT ?",
        (mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


def generate_auto_briefing(mid: int):
    """Called by background thread every BRIEFING_INTERVAL_SEC."""
    con = get_db()
    logs = con.execute(
        "SELECT level, message FROM mission_logs WHERE mission_id=? ORDER BY id DESC LIMIT 20", (mid,)
    ).fetchall()
    telem = con.execute(
        "SELECT data FROM telemetry_snapshots WHERE mission_id=? ORDER BY id DESC LIMIT 1", (mid,)
    ).fetchone()
    con.close()

    ctx_logs  = "\n".join(f"[{r['level']}] {r['message']}" for r in logs)
    ctx_telem = json.dumps(json.loads(telem["data"]), indent=2) if telem else "unavailable"

    result, err = call_anthropic({
        "system":     "You are NEXUS, an AI mission controller. Give a brief 3-sentence mission status update. Be concise and factual.",
        "max_tokens": 300,
        "messages":   [{
            "role": "user",
            "content": f"Recent logs:\n{ctx_logs}\n\nLatest telemetry:\n{ctx_telem}\n\nGive a brief mission status update.",
        }],
    })
    if err:
        print("  ! Briefing AI error:", err)
        return

    briefing = result.get("content", [{}])[0].get("text", "")
    ts = datetime.utcnow().isoformat()
    con = get_db()
    con.execute(
        "INSERT INTO mission_briefings (mission_id, timestamp, content) VALUES (?,?,?)",
        (mid, ts, briefing),
    )
    con.commit(); con.close()
    socketio.emit("briefing", {"timestamp": ts, "content": briefing})
    print("  ✓  Auto briefing generated and broadcast")


def briefing_loop():
    time.sleep(30)           # give server time to warm up
    while True:
        generate_auto_briefing(active_mission())
        time.sleep(BRIEFING_INTERVAL_SEC)


# ── NEW: Voice-ready alerts endpoint ──────────────────────────────────────────
@app.route("/api/alerts/latest", methods=["GET"])
def get_latest_alerts():
    """Returns the latest predictive/threshold alerts for voice readout."""
    return jsonify(latest_alerts[-10:])


# ── Live anomaly injection ─────────────────────────────────────────────────────
ANOMALY_LOG = {
    "thermal": ("Thermal spike injected · 3 ticks", "fa-fire"),
    "comm":    ("Comm-loss fault injected",          "fa-satellite-dish"),
    "power":   ("Power fault injected",              "fa-bolt"),
    "clear":   ("Anomalies cleared",                 "fa-broom"),
}


def explain_anomaly_async(kind: str, mid: int, anomaly_id: int):
    """Background task: ask Claude to explain a fault, then update DB + broadcast."""
    telem_ctx = ""
    con = get_db()
    row = con.execute(
        "SELECT data FROM telemetry_snapshots WHERE mission_id=? ORDER BY id DESC LIMIT 1", (mid,)
    ).fetchone()
    con.close()
    if row:
        telem_ctx = json.dumps(json.loads(row["data"]), indent=2)

    result, err = call_anthropic({
        "system":     "You are NEXUS AI, a spacecraft fault diagnostics system. Be concise (3-5 sentences).",
        "max_tokens": 400,
        "messages":   [{
            "role": "user",
            "content": (
                f"A '{kind}' fault was injected on the spacecraft.\n"
                f"Current telemetry:\n{telem_ctx}\n\n"
                "Explain: (1) likely causes, (2) immediate crew risk, (3) recommended action."
            ),
        }],
    })
    if err:
        return

    explanation = result.get("content", [{}])[0].get("text", "")
    con = get_db()
    con.execute(
        "UPDATE anomaly_events SET explanation=? WHERE id=?", (explanation, anomaly_id)
    )
    con.commit(); con.close()
    socketio.emit("anomaly_explanation", {"anomaly_id": anomaly_id, "kind": kind, "explanation": explanation})


@app.route("/api/anomaly/inject", methods=["POST"])
def inject_anomaly():
    data = request.get_json(force=True)
    kind = data.get("type", "")
    if kind not in ANOMALY_LOG:
        return jsonify({"error": "unknown anomaly type"}), 400

    with sim_lock:
        if kind == "clear":
            sim_state["injected"] = {"thermal": 0, "comm": 0, "power": 0}
        else:
            sim_state["injected"][kind] = 3

    message, icon = ANOMALY_LOG[kind]
    ts  = datetime.utcnow().isoformat()
    mid = active_mission()
    con = get_db()

    # Log entry
    cur = con.execute(
        "INSERT INTO mission_logs (mission_id, timestamp, level, message) VALUES (?,?,?,?)",
        (mid, ts, "WARN" if kind != "clear" else "INFO", message),
    )
    log_id = cur.lastrowid
    con.execute("INSERT INTO logs_fts (rowid, message) VALUES (?,?)", (log_id, message))

    # Anomaly event record
    anomaly_id = None
    if kind != "clear":
        cur2 = con.execute(
            "INSERT INTO anomaly_events (mission_id, kind, severity, start_ts) VALUES (?,?,?,?)",
            (mid, kind, "WARN", ts),
        )
        anomaly_id = cur2.lastrowid
    else:
        # Close open anomalies
        con.execute(
            "UPDATE anomaly_events SET end_ts=? WHERE mission_id=? AND end_ts IS NULL",
            (ts, mid),
        )

    con.commit(); con.close()

    socketio.emit("log",     {"id": log_id, "timestamp": ts,
                               "level": "WARN" if kind != "clear" else "INFO",
                               "message": message, "icon": icon})
    socketio.emit("anomaly", {"type": kind})

    # Fire off async AI explanation (non-blocking)
    if anomaly_id and ANTHROPIC_API_KEY:
        threading.Thread(
            target=explain_anomaly_async, args=(kind, mid, anomaly_id), daemon=True
        ).start()

    return jsonify({"status": "ok", "injected": sim_state["injected"]})


# ── Predictive z-score anomaly detection ───────────────────────────────────────
SENSOR_THRESHOLDS = {
    "temp":     (18, 92),
    "battery":  (60, 100),
    "signal":   (50, 100),
    "pressure": (94, 115),
    "cpu":      (0,  95),
    "spo2":     (94, 100),
    "heart_rate": (55, 100),
}


def check_zscore_alerts(snap: dict) -> list[dict]:
    """
    For each sensor, maintain a rolling window and flag readings that deviate
    more than ZSCORE_THRESHOLD standard deviations from the rolling mean.
    Also checks hard thresholds defined in SENSOR_THRESHOLDS.
    Returns a list of alert dicts.
    """
    alerts = []
    for sensor, value in snap.items():
        if sensor == "timestamp":
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue

        buf = rolling_buffers.setdefault(sensor, deque(maxlen=ROLLING_WINDOW))

        # Hard threshold check
        if sensor in SENSOR_THRESHOLDS:
            lo, hi = SENSOR_THRESHOLDS[sensor]
            if fval < lo or fval > hi:
                alerts.append({
                    "sensor":    sensor,
                    "value":     fval,
                    "type":      "threshold",
                    "message":   f"{sensor} out of range: {fval} (expected {lo}–{hi})",
                    "timestamp": snap.get("timestamp", datetime.utcnow().isoformat()),
                })

        # Z-score check (only once we have enough samples)
        if len(buf) >= ROLLING_WINDOW // 2:
            mean = statistics.mean(buf)
            try:
                stdev = statistics.stdev(buf)
            except statistics.StatisticsError:
                stdev = 0
            if stdev > 0:
                z = abs(fval - mean) / stdev
                if z > ZSCORE_THRESHOLD:
                    alerts.append({
                        "sensor":    sensor,
                        "value":     fval,
                        "type":      "zscore",
                        "z":         round(z, 2),
                        "message":   f"{sensor} anomaly detected (z={z:.2f}): {fval}",
                        "timestamp": snap.get("timestamp", datetime.utcnow().isoformat()),
                    })

        buf.append(fval)

    return alerts


# ── Telemetry generation ───────────────────────────────────────────────────────
def gen_telemetry() -> dict:
    with sim_lock:
        inj      = sim_state["injected"]
        battery  = random.randint(72, 100)
        fuel     = random.randint(42, 92)
        temp     = random.randint(18, 92)
        pressure = random.randint(96, 112)
        cpu      = random.randint(18, 96)
        signal   = random.randint(58, 100)

        if inj["thermal"] > 0:
            temp = random.randint(85, 98);  inj["thermal"] -= 1
        if inj["comm"]    > 0:
            signal = random.randint(20, 45); inj["comm"]   -= 1
        if inj["power"]   > 0:
            battery = random.randint(30, 55); inj["power"] -= 1

    o2          = round(random.uniform(20.5, 21.2), 1)
    co2         = random.randint(400, 480)
    hum         = random.randint(38, 55)
    cabin       = random.randint(20, 24)
    heart_rate  = random.randint(68, 88)
    solar_gen   = (
        round(random.uniform(3.8, 6.2), 1)
        if not sim_state["injected"]["power"]
        else round(random.uniform(0.5, 1.5), 1)
    )
    consumption = round(random.uniform(2.8, 4.5), 1)

    # NEW: extended crew vitals
    spo2        = round(random.uniform(95.5, 99.0), 1)
    resp_rate   = round(random.uniform(12.0, 18.0), 1)
    stress_idx  = round(random.uniform(1.0, 4.0), 1)

    return {
        "timestamp":   datetime.utcnow().isoformat(),
        "battery":     battery,
        "fuel":        fuel,
        "temp":        temp,
        "pressure":    pressure,
        "cpu":         cpu,
        "signal":      signal,
        "o2":          o2,
        "co2":         co2,
        "hum":         hum,
        "cabin":       cabin,
        "heartRate":   heart_rate,
        "solarGen":    solar_gen,
        "consumption": consumption,
        # Extended crew vitals
        "spo2":        spo2,
        "respRate":    resp_rate,
        "stressIdx":   stress_idx,
    }


def telemetry_loop():
    while True:
        payload = gen_telemetry()
        mid     = active_mission()

        # Persist
        try:
            con = get_db()
            con.execute(
                "INSERT INTO telemetry_snapshots (mission_id, timestamp, data) VALUES (?,?,?)",
                (mid, payload["timestamp"], json.dumps(payload)),
            )
            # Persist crew health separately for easy querying
            con.execute(
                "INSERT INTO crew_health (mission_id, timestamp, heart_rate, spo2, resp_rate, stress_idx) VALUES (?,?,?,?,?,?)",
                (mid, payload["timestamp"], payload["heartRate"],
                 payload["spo2"], payload["respRate"], payload["stressIdx"]),
            )
            con.commit(); con.close()
        except Exception as e:
            print("  ! telemetry persist error:", e)

        # Z-score / threshold alert check
        alerts = check_zscore_alerts(payload)
        if alerts:
            latest_alerts.extend(alerts)
            if len(latest_alerts) > 100:
                del latest_alerts[:-100]
            for alert in alerts:
                socketio.emit("alert", alert)

        socketio.emit("telemetry", payload)
        time.sleep(TELEMETRY_INTERVAL_SEC)


# ── Chat History ───────────────────────────────────────────────────────────────
@app.route("/api/chat/history", methods=["GET"])
def get_chat_history():
    session = request.args.get("session", "analyst")
    limit   = request.args.get("limit", 50, type=int)
    mid     = request.args.get("mission_id", active_mission(), type=int)
    con     = get_db()
    rows    = con.execute(
        "SELECT * FROM chat_history WHERE session=? AND mission_id=? ORDER BY id DESC LIMIT ?",
        (session, mid, limit),
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in reversed(rows)])


@app.route("/api/chat/history", methods=["DELETE"])
def clear_chat_history():
    session = request.args.get("session", "analyst")
    mid     = active_mission()
    con     = get_db()
    con.execute("DELETE FROM chat_history WHERE session=? AND mission_id=?", (session, mid))
    con.commit(); con.close()
    return jsonify({"status": "cleared"})


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status":         "online",
        "version":        "3.0.0",
        "ai":             "connected" if ANTHROPIC_API_KEY else "no_key",
        "database":       os.path.exists(DB_PATH),
        "server":         "Flask",
        "active_mission": active_mission(),
        "features": [
            "telemetry_analytics",
            "ai_anomaly_explanation",
            "log_fts_search",
            "timeline_replay",
            "crew_health",
            "multi_mission",
            "auto_briefing",
            "csv_export",
            "ai_report",
            "predictive_alerts",
            "zscore_detection",
            "retention_policy",
        ],
    })


# ── WebSocket connect ──────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    socketio.emit("log", {
        "message":   "New mission-control screen connected",
        "icon":      "fa-satellite",
        "timestamp": datetime.utcnow().isoformat(),
        "level":     "INFO",
    })


# ── Boot ───────────────────────────────────────────────────────────────────────
def open_browser():
    time.sleep(0.9)
    import webbrowser
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    init_db()
    prune_old_telemetry()    # prune on startup

    key_status = (
        "\033[92m✓ API key loaded\033[0m"
        if ANTHROPIC_API_KEY
        else "\033[91m✗ No API key — export ANTHROPIC_API_KEY=sk-ant-...\033[0m"
    )

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   NEXUS AI · Artemis-X Mission Control  v3.0            ║")
    print("  ║   Flask + Claude AI + SQLite + Enhanced Features        ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║   http://localhost:{PORT:<37}║")
    print("  ║                                                          ║")
    print("  ║   NEW ENDPOINTS                                          ║")
    print("  ║   GET  /api/telemetry/stats      Sensor analytics        ║")
    print("  ║   GET  /api/telemetry/range      Timeline replay         ║")
    print("  ║   GET  /api/logs/search?q=...    FTS log search          ║")
    print("  ║   GET  /api/crew/health          Crew vitals history      ║")
    print("  ║   GET  /api/anomalies            Anomaly history          ║")
    print("  ║   GET  /api/missions             Multi-mission CRUD       ║")
    print("  ║   GET  /api/briefings            Auto AI briefings        ║")
    print("  ║   GET  /api/export/csv           Download telemetry CSV   ║")
    print("  ║   GET  /api/export/report        AI mission report        ║")
    print("  ║   GET  /api/alerts/latest        Predictive alerts        ║")
    print("  ║                                                          ║")
    print("  ║   NEW WEBSOCKET EVENTS                                   ║")
    print("  ║   alert              Threshold / z-score alert           ║")
    print("  ║   anomaly_explanation  Claude fault diagnosis             ║")
    print("  ║   briefing           Auto AI mission briefing            ║")
    print("  ║   mission_changed    Active mission switched             ║")
    print("  ║                                                          ║")
    print("  ║   Press Ctrl-C to stop                                   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"  {key_status}")
    print()

    threading.Thread(target=open_browser,  daemon=True).start()
    threading.Thread(target=telemetry_loop, daemon=True).start()
    threading.Thread(target=briefing_loop,  daemon=True).start()
    threading.Thread(target=retention_loop, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=PORT, debug=False, allow_unsafe_werkzeug=True)
