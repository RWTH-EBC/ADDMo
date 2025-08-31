import os, json, uuid, socket, tempfile
from datetime import datetime, timedelta, timezone
import streamlit as st

DEFAULT_LEASE_FILE = os.path.join(tempfile.gettempdir(), "addmo_gui_lease.json")
LEASE_FILE = os.environ.get("ADDMO_LEASE_FILE", DEFAULT_LEASE_FILE)
LEASE_HOURS = int(os.environ.get("ADDMO_LEASE_HOURS", "12"))
IDLE_SECONDS = int(os.environ.get("ADDMO_IDLE_SECONDS", str(6 * 3600)))  # 6h idle cap
GONE_SECONDS = int(os.environ.get("ADDMO_GONE_SECONDS", "45"))          # 45 seconds after tab closes
JOB_STALE_SECONDS = int(os.environ.get("ADDMO_JOB_STALE_SECONDS", "900")) # 15 min if job stops pinging

def _now(): return datetime.now(timezone.utc)
def _iso(dt): return dt.astimezone(timezone.utc).isoformat()
def _parse(s): return datetime.fromisoformat(s)

def _read():
    try:
        with open(LEASE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return None

def _write(d):
    tmp = LEASE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(d, f)
    os.replace(tmp, LEASE_FILE)

def _remove():
    try: os.remove(LEASE_FILE)
    except FileNotFoundError: pass

def _stale(lease):
    try:
        now = _now()
        last_ui = _parse(lease.get("last_ui", lease.get("last_seen")))
        job_running = bool(lease.get("job_running", False))
        last_job = _parse(lease.get("last_job", lease.get("last_seen")))


        if not job_running:
            if (now - last_ui).total_seconds() > GONE_SECONDS:
                return True
            if (now - last_ui).total_seconds() > IDLE_SECONDS:
                return True
            return False

        if (now - last_job).total_seconds() > JOB_STALE_SECONDS:
            return True

        return False
    except Exception:
        return True


def acquire_or_block(user_hint: str = "", sid: str | None = None):
    """Call once near the top of app.py, after set_page_config. Blocks others."""
    if not sid:
        if "session_token" not in st.session_state:
            import uuid
            st.session_state.session_token = str(uuid.uuid4())
        sid = st.session_state.session_token

    # Try atomic create
    try:
        with open(LEASE_FILE, "x", encoding="utf-8") as f:
            json.dump(_new_lease(sid, user_hint), f)
        return _read()
    except FileExistsError:
        pass

    lease = _read()
    if not lease or _now() >= _parse(lease["expires_at"]) or _stale(lease):
        lease = _new_lease(sid, user_hint)
        _write(lease)
        return lease

    if lease.get("owner_session_id") == sid:
        return lease

    # Otherwise block
    exp_local = _parse(lease["expires_at"]).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    st.warning(f"GUI is currently **in use**. Available for use from **{exp_local}**.")
    st.stop()

def heartbeat(lease: dict, sid: str | None = None):
    if not lease or not sid:
        return
    cur = _read()
    if cur and cur.get("owner_session_id") == sid:
        now_iso = _iso(_now())
        cur["last_seen"] = now_iso
        cur["last_ui"] = now_iso   # NEW: UI heartbeat
        _write(cur)

def _set_job_state(sid: str, running: bool):
    cur = _read()
    if not cur or cur.get("owner_session_id") != sid:
        return
    cur["job_running"] = bool(running)
    cur["last_job"] = _iso(_now())
    _write(cur)

def job_heartbeat(sid: str):
    """Call periodically from long-running code (e.g., each training loop iteration)."""
    cur = _read()
    if not cur or cur.get("owner_session_id") != sid:
        return
    cur["last_job"] = _iso(_now())
    _write(cur)

def mark_job_start(sid: str):
    _set_job_state(sid, True)

def mark_job_done(sid: str):
    _set_job_state(sid, False)

# Optional convenience for wrapping big jobs
from contextlib import contextmanager
@contextmanager
def job_guard(sid: str):
    mark_job_start(sid)
    try:
        yield
    finally:
        mark_job_done(sid)

def release_if_owner(sid: str | None = None):
    cur = _read()
    if not cur:
        return
    is_owner = (sid is not None) and (cur.get("owner_session_id") == sid)
    is_expired = _now() >= _parse(cur["expires_at"])
    if is_owner or is_expired:
        _remove()
    else:
        st.error("You are not the owner of the current lease.")

def enforce_max_age(lease: dict):
    """Kick the owner out once 12h are up (no soft refresh)."""
    if not lease: return
    if _now() >= _parse(lease["expires_at"]):
        # Release and inform the user
        _remove()
        st.error("Your session reached the 12‑hour limit and has been closed.")
        st.stop()

def _new_lease(session_id, hint=""):
    t = _now()
    return {
        "owner_session_id": session_id,
        "owner_hint": hint,
        "created_at": _iso(t),
        "expires_at": _iso(t + timedelta(hours=LEASE_HOURS)),  # hard cap
        "last_seen": _iso(t),                                   # legacy heartbeat (kept)
        "host": socket.gethostname(),

        "last_ui": _iso(t),         # updated by heartbeat()
        "job_running": False,       # set True while a job is active
        "last_job": _iso(t),        # updated by job_heartbeat()
    }