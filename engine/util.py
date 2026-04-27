import os
import csv
from datetime import datetime


def calculateDistance(x0, y0, x1, y1):
    return abs(((x0 - x1) * (x0 - x1)) + ((y0 - y1) * (y0 - y1)))


def write_to_csv(filename, header, data, start_date_string, folder_name="result"):
    folder_path = os.path.join(folder_name, start_date_string)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = os.path.join(folder_path, filename)
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)

        writer.writerow(data)
