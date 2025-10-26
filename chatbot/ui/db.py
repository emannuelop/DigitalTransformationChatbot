from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Tuple, Optional, Dict
import json
from passlib.hash import pbkdf2_sha256 as HASHER

try:
    from passlib.hash import bcrypt as _bcrypt
except Exception:
    _bcrypt = None
try:
    from passlib.hash import bcrypt_sha256 as _bcrypt_sha256
except Exception:
    _bcrypt_sha256 = None

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return any(r[1] == column for r in cur.fetchall())


def init_db() -> None:
    with get_conn() as c:
        # users
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

        # chats
        c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at);")

        # messages
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            text TEXT NOT NULL,
            sources TEXT, -- NOVA COLUNA PARA GUARDAR FONTES
            created_at TEXT NOT NULL,
            chat_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);")

        # Garantir colunas novas
        if not _table_has_column(c, "messages", "chat_id"):
            c.execute("ALTER TABLE messages ADD COLUMN chat_id INTEGER;")
        if not _table_has_column(c, "messages", "sources"):
            c.execute("ALTER TABLE messages ADD COLUMN sources TEXT;")

        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);")

        # Migrar mensagens antigas
        cur_users = c.execute("SELECT DISTINCT user_id FROM messages WHERE chat_id IS NULL;").fetchall()
        for (uid,) in cur_users:
            row = c.execute(
                "SELECT id FROM chats WHERE user_id=? AND title=? LIMIT 1;",
                (uid, "Histórico antigo")
            ).fetchone()
            if row:
                chat_id = row[0]
            else:
                ts = _now()
                chat_id = c.execute(
                    "INSERT INTO chats(user_id, title, created_at, updated_at) VALUES (?,?,?,?);",
                    (uid, "Histórico antigo", ts, ts)
                ).lastrowid
            c.execute("UPDATE messages SET chat_id=? WHERE user_id=? AND chat_id IS NULL;", (chat_id, uid))


# ---------- Usuários ----------
def email_exists(email: str) -> bool:
    with get_conn() as c:
        return c.execute("SELECT 1 FROM users WHERE email=? LIMIT 1;", (email.strip().lower(),)).fetchone() is not None


def get_user(email: str) -> Optional[Dict]:
    with get_conn() as c:
        r = c.execute(
            "SELECT id, name, email, password_hash, created_at FROM users WHERE email=?;",
            (email.strip().lower(),)
        ).fetchone()
        if not r:
            return None
        return {"id": r[0], "name": r[1], "email": r[2], "password_hash": r[3], "created_at": r[4]}


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_conn() as c:
        r = c.execute(
            "SELECT id, name, email, password_hash, created_at FROM users WHERE id=?;",
            (user_id,)
        ).fetchone()
        if not r:
            return None
        return {"id": r[0], "name": r[1], "email": r[2], "password_hash": r[3], "created_at": r[4]}


def create_user(name: str, email: str, password: str) -> int:
    email = email.strip().lower()
    if email_exists(email):
        raise ValueError("Já existe uma conta com este e-mail.")
    password_hash = HASHER.hash(password)
    with get_conn() as c:
        return c.execute(
            "INSERT INTO users(name, email, password_hash, created_at) VALUES (?,?,?,?);",
            (name.strip(), email, password_hash, _now())
        ).lastrowid


def authenticate(email: str, password: str) -> Optional[Dict]:
    user = get_user(email)
    if not user:
        return None
    stored = user["password_hash"]
    try:
        if stored.startswith("$pbkdf2-sha256$") and HASHER.verify(password, stored):
            return {"id": user["id"], "name": user["name"], "email": user["email"]}
    except Exception:
        pass
    if _bcrypt_sha256 is not None:
        try:
            if _bcrypt_sha256.identify(stored) and _bcrypt_sha256.verify(password, stored):
                return {"id": user["id"], "name": user["name"], "email": user["email"]}
        except Exception:
            pass
    if _bcrypt is not None:
        try:
            if _bcrypt.identify(stored) and _bcrypt.verify(password, stored):
                return {"id": user["id"], "name": user["name"], "email": user["email"]}
        except Exception:
            pass
    return None


def update_user_name(user_id: int, new_name: str) -> None:
    if not new_name.strip():
        raise ValueError("Nome não pode ser vazio.")
    with get_conn() as c:
        c.execute("UPDATE users SET name=? WHERE id=?;", (new_name.strip(), user_id))


def update_user_password(user_id: int, current_password: str, new_password: str) -> None:
    user = get_user_by_id(user_id)
    if not user:
        raise ValueError("Usuário não encontrado.")
    stored = user["password_hash"]
    ok = False
    try:
        if stored.startswith("$pbkdf2-sha256$") and HASHER.verify(current_password, stored):
            ok = True
    except Exception:
        pass
    if not ok and _bcrypt_sha256 is not None:
        try:
            if _bcrypt_sha256.identify(stored) and _bcrypt_sha256.verify(current_password, stored):
                ok = True
        except Exception:
            pass
    if not ok and _bcrypt is not None:
        try:
            if _bcrypt.identify(stored) and _bcrypt.verify(current_password, stored):
                ok = True
        except Exception:
            pass
    if not ok:
        raise ValueError("Senha atual incorreta.")
    new_hash = HASHER.hash(new_password)
    with get_conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE id=?;", (new_hash, user_id))


# ---------- Chats & Mensagens ----------
def create_chat(user_id: int, title: str = "Novo chat") -> int:
    ts = _now()
    with get_conn() as c:
        return c.execute(
            "INSERT INTO chats(user_id, title, created_at, updated_at) VALUES (?,?,?,?);",
            (user_id, title.strip() or "Novo chat", ts, ts)
        ).lastrowid


def list_chats(user_id: int, limit: int = 100) -> List[Dict]:
    with get_conn() as c:
        cur = c.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE user_id=? ORDER BY updated_at DESC LIMIT ?;",
            (user_id, limit)
        )
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in cur.fetchall()]


def rename_chat(user_id: int, chat_id: int, new_title: str) -> None:
    with get_conn() as c:
        res = c.execute(
            "UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?;",
            (new_title.strip() or "Sem título", _now(), chat_id, user_id)
        )
        if res.rowcount == 0:
            raise ValueError("Chat não encontrado.")


def delete_chat(user_id: int, chat_id: int) -> None:
    with get_conn() as c:
        res = c.execute("DELETE FROM chats WHERE id=? AND user_id=?;", (chat_id, user_id))
        if res.rowcount == 0:
            raise ValueError("Chat não encontrado.")


def save_message(user_id: int, role: str, text: str, chat_id: Optional[int] = None, urls: list[str] | None = None) -> int:
    with get_conn() as c:
        sources = json.dumps(urls or [])
        mid = c.execute(
            "INSERT INTO messages(user_id, role, text, sources, created_at, chat_id) VALUES (?,?,?,?,?,?);",
            (user_id, role, text, sources, _now(), chat_id)
        ).lastrowid
        if chat_id:
            c.execute("UPDATE chats SET updated_at=? WHERE id=?;", (_now(), chat_id))
        return mid


def load_history(user_id: int, limit: int = 500, chat_id: Optional[int] = None) -> List[Tuple[str, str, list[str], str]]:
    with get_conn() as c:
        if chat_id is None:
            cur = c.execute(
                "SELECT role, text, sources, created_at FROM messages WHERE user_id=? ORDER BY created_at ASC LIMIT ?;",
                (user_id, limit)
            )
        else:
            cur = c.execute(
                "SELECT role, text, sources, created_at FROM messages WHERE user_id=? AND chat_id=? ORDER BY created_at ASC LIMIT ?;",
                (user_id, chat_id, limit)
            )
        return [(r[0], r[1], json.loads(r[2] or "[]"), r[3]) for r in cur.fetchall()]


def clear_history(user_id: int, chat_id: Optional[int] = None) -> None:
    with get_conn() as c:
        if chat_id is None:
            c.execute("DELETE FROM messages WHERE user_id=?;", (user_id,))
        else:
            c.execute("DELETE FROM messages WHERE user_id=? AND chat_id=?;", (user_id, chat_id))
