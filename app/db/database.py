import os
from psycopg_pool import ConnectionPool
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


def configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()


def create_pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=os.environ["DATABASE_URL"],
        min_size=1,
        max_size=5,
        timeout=10,
        kwargs={
            "row_factory": dict_row,
        },
        configure=configure_connection,
        open=False,
    )


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    )

    register_vector(conn)

    return conn