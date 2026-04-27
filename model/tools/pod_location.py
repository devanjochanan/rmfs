import sqlite3

TS = None

def clear_pod_locations(db_path="warehouse.db"):
    """
    Delete all rows from the pod_location table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM pod_location_{TS}")

    conn.commit()
    conn.close()
    print("All pod locations have been cleared.")

def initialize_pod_location_table(timestamp: str, db_path="warehouse.db"):
    global TS
    TS = timestamp   
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pod_location_{TS} (
            id TEXT PRIMARY KEY,
            x INTEGER,
            y INTEGER
        )
    """)

    conn.commit()
    conn.close()
    print("pod_location table initialized.")

def upsert_pod_location(pod_id: str, x: int, y: int, db_path="warehouse.db"):
    """
    Insert or update a pod's (x, y) location.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO pod_location_{TS} (id, x, y)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET x=excluded.x, y=excluded.y
    """, (pod_id, x, y))

    conn.commit()
    conn.close()
    print(f"Pod {pod_id} location set to ({x}, {y}).")

def get_pod_location(pod_id: str, db_path="warehouse.db"):
    """
    Retrieve (x, y) location of a pod by ID.
    Returns a tuple (x, y) or None if not found.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT x, y FROM pod_location_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        print(f"Pod {pod_id} is at location {result}")
        return result
    else:
        print(f"Pod {pod_id} not found.")
        return None

# Example usage
if __name__ == "__main__":
    initialize_pod_location_table()
    upsert_pod_location("POD_001", 10, 5)
    upsert_pod_location("POD_002", 3, 7)

    get_pod_location("POD_001")
    get_pod_location("POD_003")  # Not found case
