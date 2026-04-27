import os
import csv

from typing import Optional

def write_record_to(log_file: str, row: list, header: Optional[list] = None):
    return
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if header:
            if not file_exists:
                writer.writerow(header)
        # Write the data
        writer.writerow(row)