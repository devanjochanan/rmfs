import sqlite3

TS = None  # Same global timestamp

def initialize_pre_assign_table(timestamp: str, db_path="warehouse.db"):
    """
    Create the pre_assign table if it doesn't exist.
    """
    global TS
    TS = timestamp

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pre_assign_{TS} (
            time REAL,
            current TEXT,
            order_id TEXT,
            score REAL,
            bestpicker TEXT,
            bestscore REAL
        )
    """)

    conn.commit()
    conn.close()
    print("pre_assign table initialized.")

def clear_pre_assign_table(db_path="warehouse.db"):
    """
    Delete all rows from the pre_assign table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM pre_assign_{TS}")
    conn.commit()
    conn.close()
    print("All pre_assign records have been cleared.")

def insert_pre_assign(
        time: float,
        current: str,
        order: str,
        score: float,
        bestpicker: str,
        bestscore: float,
        db_path: str = "warehouse.db"):
    """
    Insert a new pre_assign record.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO pre_assign_{TS} 
        (time, current, order_id, score, bestpicker, bestscore)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time, current, order, score, bestpicker, bestscore))

    conn.commit()
    conn.close()
    print(f"Inserted pre_assign record: current={current}, order_id={order}, score={score}")
