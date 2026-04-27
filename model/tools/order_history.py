import sqlite3

TS = None

def clear_order_history(db_path="warehouse.db"):
    """
    Delete all rows from the order_history table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM order_history_{TS}")

    conn.commit()
    conn.close()
    print("All orders have been cleared.")

def initialize_order_history_table(timestamp: str, db_path="warehouse.db"):
    global TS
    TS = timestamp

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS order_history_{TS} (
            order_id TEXT PRIMARY KEY,
            arrival_time REAL,
            assigned_station TEXT,
            order_assigned_time REAL,
            order_finish_time REAL
        )
    """)

    conn.commit()
    conn.close()
    print("order_history table initialized.")

def upsert_order_history(
        order_id: str, 
        arrival_time: float = None, 
        assigned_station: str = None,
        order_assigned_time: float = None,
        order_finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Insert a new order or update an existing one in the order_history table.
    Only non-None fields are updated.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the order already exists
    cursor.execute(f"SELECT * FROM order_history_{TS} WHERE order_id = ?", (str(order_id),))
    existing = cursor.fetchone()

    if existing:
        # Build dynamic update query based on non-None fields
        fields = []
        values = []

        if arrival_time is not None:
            fields.append("arrival_time = ?")
            values.append(arrival_time)
        if assigned_station is not None:
            fields.append("assigned_station = ?")
            values.append(assigned_station)
        if order_assigned_time is not None:
            fields.append("order_assigned_time = ?")
            values.append(order_assigned_time)
        if order_finish_time is not None:
            fields.append("order_finish_time = ?")
            values.append(order_finish_time)

        if fields:
            query = f"UPDATE order_history_{TS} SET {', '.join(fields)} WHERE order_id = ?"
            values.append(str(order_id))
            cursor.execute(query, tuple(values))
    else:
        # Insert new row
        cursor.execute(f"""
            INSERT INTO order_history_{TS} (order_id, arrival_time, assigned_station, order_assigned_time, order_finish_time)
            VALUES (?, ?, ?, ?, ?)
        """, (str(order_id), arrival_time, assigned_station, order_assigned_time, order_finish_time))

    conn.commit()
    conn.close()
    print(f"Order {order_id} upserted.")

def get_order_history(order_id: str = None, db_path: str = "warehouse.db"):
    """
    Retrieve order history.
    - If order_id is provided, returns a single matching order (or None).
    - If order_id is None, returns all orders as a list of dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enables dictionary-like access
    cursor = conn.cursor()

    if order_id:
        cursor.execute(f"SELECT * FROM order_history_{TS} WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        result = dict(row) if row else None
    else:
        cursor.execute(f"SELECT * FROM order_history_{TS} ORDER BY arrival_time")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]

    conn.close()
    return result

# Example usage
if __name__ == "__main__":
    ...
