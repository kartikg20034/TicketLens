#!/usr/bin/env python3
"""Read-only connectivity/schema check for the configured Supabase database."""

from __future__ import annotations

import os
import re
import sys
from urllib.parse import parse_qsl, unquote, urlsplit


def redact_error(error: BaseException, dsn: str | None) -> str:
    message = str(error)
    if dsn:
        message = message.replace(dsn, "[redacted connection string]")
        try:
            parsed = urlsplit(dsn)
        except ValueError:
            parsed = None
        if parsed is not None:
            # Exceptions sometimes repeat only the hostname or endpoint rather
            # than the complete URI. Treat every URI component as sensitive.
            if parsed.netloc:
                message = message.replace(parsed.netloc, "[redacted endpoint]")
            if parsed.hostname:
                message = re.sub(re.escape(parsed.hostname), "[redacted host]", message, flags=re.IGNORECASE)
            if parsed.path and parsed.path != "/":
                message = message.replace(parsed.path.lstrip("/"), "[redacted database]")
            for _key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if value:
                    message = message.replace(value, "[redacted]")
            password = parsed.password
        else:
            password = None
        if password:
            for secret in {password, unquote(password)}:
                message = message.replace(secret, "[redacted]")
        message = re.sub(
            r"(?i)(postgres(?:ql)?://)[^@\s]+@",
            r"\1[redacted]@",
            message,
        )
        message = re.sub(r"(?i)(password\s*=\s*)[^\s]+", r"\1[redacted]", message)
    return message or error.__class__.__name__


def emit(success: bool, database: str = "unavailable", user: str = "unavailable",
         teams: str = "unavailable", tickets: str = "unavailable",
         embeddings: str = "unavailable", error: str | None = None) -> None:
    print(f"Connection succeeded: {'yes' if success else 'no'}")
    print(f"Current database name: {database}")
    print(f"Current database user: {user}")
    print(f"Rows in public.teams: {teams}")
    print(f"public.tickets exists: {tickets}")
    print(f"public.ticket_embeddings exists: {embeddings}")
    if error:
        print(f"Error: {error}")


def main() -> int:
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        emit(False, error="SUPABASE_DB_URL is not set in this process environment.")
        return 1

    try:
        import psycopg
    except ImportError as error:
        emit(False, error="The psycopg PostgreSQL driver is not installed.")
        return 1

    conn = None
    connected = False
    database = user = "unavailable"
    teams_count = "unavailable"
    tickets_exists = embeddings_exists = "unavailable"
    try:
        conn = psycopg.connect(dsn, connect_timeout=10)
        connected = True
        with conn.cursor() as cur:
            # Make this session read-only before running any requested queries.
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SELECT current_database(), current_user;")
            database, user = cur.fetchone()
            cur.execute("SELECT to_regclass('public.tickets'), to_regclass('public.ticket_embeddings'), to_regclass('public.teams');")
            tickets_reg, embeddings_reg, teams_reg = cur.fetchone()
            tickets_exists = "yes" if tickets_reg is not None else "no"
            embeddings_exists = "yes" if embeddings_reg is not None else "no"
            teams_count = "unavailable (public.teams does not exist)"
            if teams_reg is not None:
                cur.execute("SELECT count(*) FROM public.teams;")
                teams_count = str(cur.fetchone()[0])
        emit(True, str(database), str(user), teams_count, tickets_exists, embeddings_exists)
        return 0
    except Exception as error:
        emit(connected, str(database), str(user), teams_count, tickets_exists,
             embeddings_exists, redact_error(error, dsn))
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
