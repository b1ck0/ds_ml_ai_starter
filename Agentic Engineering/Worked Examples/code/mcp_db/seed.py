"""Seed a small SQLite database for the MCP database query layer example.

Deterministic: this script drops and recreates the tables every time it runs, so
the example queries in the chapter always see the same rows.

Run:
    .venv-agent/Scripts/python.exe seed.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "mcp_demo.db"

CUSTOMERS = [
    (1, "Ana Petrova", "ana.petrova@example.com", "BG"),
    (2, "Marco Rossi", "marco.rossi@example.com", "IT"),
    (3, "Yuki Tanaka", "yuki.tanaka@example.com", "JP"),
    (4, "Lena Novak", "lena.novak@example.com", "CZ"),
    (5, "Sam O'Brien", "sam.obrien@example.com", "IE"),
]

ORDERS = [
    (1, 1, "USB-C Hub", 29.99, "shipped"),
    (2, 1, "Mechanical Keyboard", 89.00, "shipped"),
    (3, 2, "Espresso Machine", 249.50, "pending"),
    (4, 3, "Standing Desk", 410.00, "shipped"),
    (5, 3, "Monitor Arm", 65.20, "cancelled"),
    (6, 4, "Noise-Cancelling Headphones", 199.99, "pending"),
    (7, 5, "Webcam", 74.99, "shipped"),
    (8, 2, "Office Chair", 320.00, "shipped"),
]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                id      INTEGER PRIMARY KEY,
                name    TEXT NOT NULL,
                email   TEXT NOT NULL,
                country TEXT NOT NULL
            );

            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                item        TEXT NOT NULL,
                amount      REAL NOT NULL,
                status      TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO customers (id, name, email, country) VALUES (?, ?, ?, ?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, item, amount, status) VALUES (?, ?, ?, ?, ?)",
            ORDERS,
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Seeded {DB_PATH.name} with {len(CUSTOMERS)} customers and {len(ORDERS)} orders.")


if __name__ == "__main__":
    main()
