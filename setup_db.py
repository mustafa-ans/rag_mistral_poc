# setup_db.py
# One-command setup for the pgvector-backed FAQ store.
#
#   python setup_db.py
#
# It will:
#   1. create the target database (rag_poc_db) if it doesn't exist,
#   2. enable pgvector and create the `faq` table + index,
#   3. embed every FAQ question locally and load it into the table.
#
# Connection comes from DATABASE_URL in your .env, e.g.
#   DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/rag_poc_db

import sys
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
import psycopg
from psycopg import sql

import db


def admin_url_and_dbname(app_url: str):
    """Derive an admin connection URL (pointing at the default 'postgres'
    database) plus the target DB name from the app's DATABASE_URL. CREATE
    DATABASE cannot run while connected to the database being created, so we
    connect to 'postgres' for that step."""
    parsed = urlparse(app_url)
    dbname = parsed.path.lstrip("/")
    if not dbname:
        raise RuntimeError("DATABASE_URL has no database name in its path.")
    admin = parsed._replace(path="/postgres")
    return urlunparse(admin), dbname


def create_database_if_missing(admin_url: str, dbname: str) -> None:
    # autocommit: CREATE DATABASE cannot run inside a transaction block.
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
            if cur.fetchone():
                print(f"Database '{dbname}' already exists.")
            else:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                print(f"Created database '{dbname}'.")


def main() -> None:
    load_dotenv()

    try:
        app_url = db.get_database_url()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    admin_url, dbname = admin_url_and_dbname(app_url)
    print(f"Target database: {dbname}")

    try:
        create_database_if_missing(admin_url, dbname)
    except psycopg.OperationalError as e:
        print(f"Could not connect to Postgres: {e}")
        print("Check the host/user/password in DATABASE_URL and that Postgres is running.")
        sys.exit(1)

    try:
        with db.connect(app_url) as conn:
            db.init_schema(conn)          # may raise if pgvector isn't installed
            n = db.ingest_faq(conn)
            total = db.faq_count(conn)
            print(f"Ingested {n} FAQ rows. Table now holds {total} rows.")
            print("Setup complete. Run:  python main.py")
    except RuntimeError as e:             # pgvector-not-installed guidance from init_schema
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
