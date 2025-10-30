
# /content/drive/MyDrive/Project_PFE/app/database.py

import os
import uuid
import mimetypes
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client
except Exception:
    create_client = None


class Database:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            self.client = None
            print("SUPABASE_URL/KEY missing in .env")
            return

        if create_client is None:
            raise RuntimeError("Install supabase first: pip install supabase")

        self.client = create_client(url, key)
        print("Supabase client ready")

    def save_analysis(self, d: dict):
        if not self.client:
            return None

        record = {
            "id": str(uuid.uuid4()),
            "user_id": d.get("user_id", "anonymous"),
            "file_name": d.get("file_name", ""),
            "file_type": d.get("file_type", ""),
            "vehicle_count": int(d.get("vehicle_count", 0)),
            "occupancy": float(d.get("occupancy", 0)),
            "flow_rate_vph": float(d.get("flow_rate_vph", 0)),
            "congestion_level": int(d.get("congestion_level", 0)),
            "processing_duration": float(d.get("processing_duration", 0)),
            "created_at": datetime.utcnow().isoformat(),
        }

        res = self.client.table("analyses").insert(record).execute()
        return record["id"] if getattr(res, "data", None) else None

    def get_history(self, user_id: str = "anonymous", limit: int = 50):
        if not self.client:
            return []

        res = (
            self.client.table("analyses")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return getattr(res, "data", []) or []

    def save_realtime_metrics(self, metrics_list: list, stream_url: str, user_id: str = "anonymous", session_id: str = None):
        """Save time-series metrics from real-time stream"""
        if not self.client or not metrics_list:
            print("[DB] No client or empty metrics")
            return None

        # Use provided session_id or create new one
        if session_id is None:
            session_id = str(uuid.uuid4())
            print(f"[DB] Created new session: {session_id}")
        else:
            print(f"[DB] Reusing session: {session_id}")

        records = []
        for m in metrics_list:
            record = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "user_id": user_id,
                "stream_url": stream_url,
                "timestamp": float(m["timestamp"]),
                "vehicle_count": int(m["vehicles"]),
                "active_lanes": int(m["active_lanes"]),
                "occupancy": float(m["occupancy"]),
                "crosswalk_blocked": float(m.get("crosswalk_blocked", 0)),
                "flow_rate_vph": float(m["flow_rate_vph"]),
                "volume_capacity_ratio": float(m.get("volume_capacity_ratio", 0)),
                "congestion_level": int(m["congestion_level"]),
                "created_at": datetime.utcnow().isoformat(),
            }
            records.append(record)
            print(f"[DB] Prepared record: session={session_id}, timestamp={record['timestamp']}, vehicles={record['vehicle_count']}")

        # Insert records
        try:
            result = self.client.table("realtime_metrics").insert(records).execute()
            print(f"[DB] ✓ Inserted {len(records)} records successfully")
            print(f"[DB] Result data: {getattr(result, 'data', 'no data')}")
            return session_id
        except Exception as e:
            print(f"[DB] ✗ Insert failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_realtime_sessions(self, user_id: str = "anonymous", limit: int = 20):
        """Get list of real-time monitoring sessions"""
        if not self.client:
            print("[DB] No client for get_realtime_sessions")
            return []

        try:
            res = (
                self.client.table("realtime_metrics")
                .select("session_id, stream_url, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(200)  # Get more rows to group
                .execute()
            )

            rows = getattr(res, "data", []) or []
            print(f"[DB] Found {len(rows)} total metrics rows")

            # Group by session
            sessions = {}
            for row in rows:
                sid = row["session_id"]
                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "stream_url": row["stream_url"],
                        "started_at": row["created_at"],
                        "snapshot_count": 0
                    }
                sessions[sid]["snapshot_count"] += 1

            result = list(sessions.values())[:limit]
            print(f"[DB] Returning {len(result)} sessions")
            return result

        except Exception as e:
            print(f"[DB] get_realtime_sessions error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_session_metrics(self, session_id: str):
        """Get all metrics for a specific session"""
        if not self.client:
            print("[DB] No client for get_session_metrics")
            return []

        try:
            res = (
                self.client.table("realtime_metrics")
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
                .execute()
            )

            metrics = getattr(res, "data", []) or []
            print(f"[DB] Found {len(metrics)} metrics for session {session_id}")
            return metrics

        except Exception as e:
            print(f"[DB] get_session_metrics error: {e}")
            import traceback
            traceback.print_exc()
            return []


_db_singleton = None


def get_database():
    global _db_singleton
    if _db_singleton is None:
        try:
            _db_singleton = Database()
        except Exception as e:
            print(f"Database init error: {e}")
            _db_singleton = Database.__new__(Database)
            _db_singleton.client = None
    return _db_singleton


__all__ = ["Database", "get_database"]


