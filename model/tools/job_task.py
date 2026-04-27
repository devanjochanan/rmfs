import sqlite3

TS = None

def initialize_job_task_table(timestamp: str, db_path="warehouse.db"):
    """
    Create the job_task table if it doesn't exist.
    """
    global TS
    TS = timestamp

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS job_task_{TS} (
            pod_id INTEGER,
            order_id INTEGER,
            sku INTEGER,
            qty INTEGER,
            assigned_station TEXT,
            pod_assigned_time REAL,
            status TEXT,
            finish_time REAL,
            PRIMARY KEY (pod_id, order_id, sku, qty)
        )
    """)

    conn.commit()
    conn.close()
    print("job_task table initialized.")

def clear_job_task_table(db_path="warehouse.db"):
    """
    Delete all rows from the job_task table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM job_task_{TS}")

    conn.commit()
    conn.close()
    print("All job tasks have been cleared.")

def upsert_job_task(
        pod_id: int,
        order_id: int,
        sku: int,
        qty: int,
        assigned_station: str = None,
        pod_assigned_time: float = None,
        status: str = None,
        finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Insert or update a job task based on (pod_id, order_id, sku, qty) as composite key.
    Only non-None fields will be updated on conflict.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the entry exists
    cursor.execute(f"""
        SELECT * FROM job_task_{TS}
        WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
    """, (pod_id, order_id, sku, qty))
    existing = cursor.fetchone()

    if existing:
        # Prepare dynamic update
        fields = []
        values = []

        if assigned_station is not None:
            fields.append("assigned_station = ?")
            values.append(assigned_station)
        if pod_assigned_time is not None:
            fields.append("pod_assigned_time = ?")
            values.append(pod_assigned_time)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if finish_time is not None:
            fields.append("finish_time = ?")
            values.append(finish_time)

        if fields:
            query = f"""
                UPDATE job_task_{TS} SET {', '.join(fields)}
                WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
            """
            values.extend([pod_id, order_id, sku, qty])
            cursor.execute(query, tuple(values))
    else:
        # Insert new record
        cursor.execute(f"""
            INSERT INTO job_task_{TS} (pod_id, order_id, sku, qty, assigned_station, pod_assigned_time, status, finish_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pod_id, order_id, sku, qty, assigned_station, pod_assigned_time, status, finish_time))

    conn.commit()
    conn.close()
    print(f"Job task ({pod_id}, {order_id}, {sku}, {qty}) upserted.")

def update_job_task(
        pod_id: int,
        order_id: int,
        sku: int,
        qty: int,
        assigned_station: str = None,
        pod_assigned_time: float = None,
        status: str = None,
        finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Insert or update a job task based on (pod_id, order_id, sku, qty) as composite key.
    Only non-None fields will be updated on conflict.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare dynamic update
    fields = []
    values = []

    if assigned_station is not None:
        fields.append("assigned_station = ?")
        values.append(assigned_station)
    if pod_assigned_time is not None:
        fields.append("pod_assigned_time = ?")
        values.append(pod_assigned_time)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if finish_time is not None:
        fields.append("finish_time = ?")
        values.append(finish_time)

    if fields:
        query = f"""
            UPDATE job_task_{TS} SET {', '.join(fields)}
            WHERE pod_id = ? AND order_id = ? AND sku = ? AND qty = ?
        """
        values.extend([pod_id, order_id, sku, qty])
        cursor.execute(query, tuple(values))
    
    conn.commit()
    conn.close()
    print(f"Job task ({pod_id}, {order_id}, {sku}, {qty}) updated.")

def get_job_task(pod_id: int = None, order_id: str = None, db_path: str = "warehouse.db"):
    """
    Retrieve job task history.
    - If pod_id and/or order_id is provided, filters accordingly.
    - If neither is provided, returns all job tasks.
    Returns a list of dictionaries.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = f"SELECT * FROM job_task_{TS}"
    conditions = []
    values = []

    if pod_id is not None:
        conditions.append("pod_id = ?")
        values.append(pod_id)
    if order_id is not None:
        conditions.append("order_id = ?")
        values.append(order_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY pod_assigned_time"

    cursor.execute(query, tuple(values))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]