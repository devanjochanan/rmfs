import sqlite3

TS = None

def initialize_pod_info_table(timestamp: str, db_path="warehouse.db"):
    global TS
    TS = timestamp   

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pod_info_{TS} (
            id TEXT PRIMARY KEY,
            x REAL,
            y REAL,
            is_idle INTEGER
        )
    """)

    conn.commit()
    conn.close()
    print("pod_info table initialized.")

def clear_pod_info(db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM pod_info_{TS}")

    conn.commit()
    conn.close()
    print("All pod info rows have been cleared.")

def upsert_pod_location(pod_id: str, x: float, y: float, db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO pod_info_{TS} (id, x, y)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET x=excluded.x, y=excluded.y
    """, (pod_id, x, y))

    conn.commit()
    conn.close()
    print(f"Pod {pod_id} location set to ({x}, {y}).")

def upsert_pod_idle(pod_id: str, is_idle: bool, db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        INSERT INTO pod_info_{TS} (id, is_idle)
        VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET is_idle=excluded.is_idle
    """, (pod_id, int(is_idle)))

    conn.commit()
    conn.close()
    print(f"Pod {pod_id} idle status set to {is_idle}.")

def get_pod_info(pod_id: str, db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT x, y, is_idle FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        print(f"Pod {pod_id} info: location=({result[0]}, {result[1]}), is_idle={bool(result[2])}")
        return result
    else:
        print(f"Pod {pod_id} not found.")
        return None

def get_pod_location(pod_id: str, db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT x, y FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        print(f"Pod {pod_id} is at location ({result[0]}, {result[1]})")
        return result
    else:
        print(f"Pod {pod_id} not found.")
        return None

def get_pod_idle(pod_id: str, db_path="warehouse.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT is_idle FROM pod_info_{TS} WHERE id = ?", (pod_id,))
    result = cursor.fetchone()
    conn.close()

    if result is not None:
        print(f"Pod {pod_id} is_idle = {bool(result[0])}")
        return bool(result[0])
    else:
        print(f"Pod {pod_id} not found.")
        return None
