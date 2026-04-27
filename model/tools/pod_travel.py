import sqlite3

TS = None

def initialize_pod_travel_table(timestamp: str, db_path="warehouse.db"):
    """
    Create the pod_travel table if it doesn't exist.
    """
    global TS
    TS = timestamp

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS pod_travel_{TS} (
            job_id TEXT,
            robot_id INTEGER,
            pod_id INTEGER,
            task TEXT,
            from_position TEXT,
            to_position TEXT,
            start_time REAL,
            finish_time REAL,
            PRIMARY KEY (job_id, robot_id, pod_id, task)
        )
    """)

    conn.commit()
    conn.close()
    print("pod_travel table initialized.")

def clear_pod_travel(db_path="warehouse.db"):
    """
    Delete all rows from the pod_travel table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"DELETE FROM pod_travel_{TS}")

    conn.commit()
    conn.close()
    print("All pod travel records have been cleared.")

def upsert_pod_travel(
        job_id: str,
        robot_id: int,
        pod_id: int,
        task: str,
        from_position: str = None,
        to_position: str = None,
        start_time: float = None,
        finish_time: float = None,
        db_path: str = "warehouse.db"):
    """
    Upsert a pod travel record.
    Updates only provided fields. Identified by (job_id, robot_id, pod_id, task).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # print(f"[DEBUG] job_id={job_id} ({type(job_id)}), robot_id={robot_id} ({type(robot_id)}), pod_id={pod_id} ({type(pod_id)}), task={task} ({type(task)})")
    # print(f"[DEBUG] from_position={from_position} ({type(from_position)}), to_position={to_position} ({type(to_position)}), start_time={start_time} ({type(start_time)}), finish_time={finish_time} ({type(finish_time)})")
    # First, check if record exists
    cursor.execute(f"""
        SELECT * FROM pod_travel_{TS} WHERE job_id = ? AND robot_id = ? AND pod_id = ? AND task = ?
    """, (job_id, robot_id, pod_id, task))
    existing = cursor.fetchone()

    if existing:
        # Prepare update clause only for non-None fields
        updates = []
        params = []

        if from_position is not None:
            updates.append("from_position = ?")
            params.append(from_position)
        if to_position is not None:
            updates.append("to_position = ?")
            params.append(to_position)
        if start_time is not None:
            updates.append("start_time = ?")
            params.append(start_time)
        if finish_time is not None:
            updates.append("finish_time = ?")
            params.append(finish_time)

        if updates:
            query = f"""
                UPDATE pod_travel_{TS}
                SET {', '.join(updates)}
                WHERE job_id = ? AND robot_id = ? AND pod_id = ? AND task = ?
            """
            params.extend([job_id, robot_id, pod_id, task])
            cursor.execute(query, tuple(params))
            print(f"Updated pod travel for pod {pod_id} (job {job_id}, robot {robot_id}, task {task}).")
        else:
            print("No fields to update.")

    else:
        # Insert new
        cursor.execute(f"""
            INSERT INTO pod_travel_{TS} 
            (job_id, robot_id, pod_id, task, from_position, to_position, start_time, finish_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, robot_id, pod_id, task,
            from_position, to_position, start_time, finish_time
        ))
        print(f"Inserted new pod travel record for pod {pod_id} (job {job_id}, robot {robot_id}, task {task}).")

    conn.commit()
    conn.close()

def get_pod_travel(pod_id: int = None, db_path: str = "warehouse.db"):
    """
    Retrieve pod travel history.
    - If pod_id is provided, returns all matching rows.
    - If pod_id is None, returns all rows.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if pod_id is not None:
        cursor.execute(f"SELECT * FROM pod_travel_{TS} WHERE pod_id = ? ORDER BY start_time", (pod_id,))
    else:
        cursor.execute(f"SELECT * FROM pod_travel_{TS} ORDER BY start_time")

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
