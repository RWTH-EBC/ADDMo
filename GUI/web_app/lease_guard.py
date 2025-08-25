import os, json, uuid, socket, tempfile
from datetime import datetime, timedelta, timezone
import streamlit as st

DEFAULT_LEASE_FILE = os.path.join(tempfile.gettempdir(), "addmo_gui_lease.json")
LEASE_FILE = os.environ.get("ADDMO_LEASE_FILE", DEFAULT_LEASE_FILE)
LEASE_HOURS = int(os.environ.get("ADDMO_LEASE_HOURS", "12"))
STALE_SECONDS = int(os.environ.get("ADDMO_LEASE_STALE_SECONDS", str(6 * 3600)))

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

def _new_lease(session_id, hint=""):
    t = _now()
    return {
        "owner_session_id": session_id,
        "owner_hint": hint,
        "created_at": _iso(t),
        "expires_at": _iso(t + timedelta(hours=LEASE_HOURS)),  # hard cap
        "last_seen": _iso(t),                                   # heartbeat
        "host": socket.gethostname(),
    }

def _stale(lease):
    try:
        return (_now() - _parse(lease["last_seen"])).total_seconds() > STALE_SECONDS
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
    st.warning(f"GUI is currently **in use**. Lease expires at **{exp_local}**.")
    st.stop()

def heartbeat(lease: dict, sid: str | None = None):
    if not lease or not sid:
        return
    cur = _read()
    if cur and cur.get("owner_session_id") == sid:
        cur["last_seen"] = _iso(_now())
        _write(cur)

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

