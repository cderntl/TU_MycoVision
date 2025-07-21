import os
import json
from collections import defaultdict
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def count_objects_in_json_format(annotation_dir):
    """
    Counts the number of objects belonging to each label in a dataset with annotations in JSON format.

    Parameters:
    annotation_dir (str): Path to the directory containing JSON annotation files.

    Returns:
    dict: A dictionary where keys are labels and values are the counts of objects in each label.
    """
    label_counts = defaultdict(int)

    # Loop through each annotation file in the directory
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.json'):
            file_path = os.path.join(annotation_dir, annotation_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
                for shape in data['shapes']:
                    label = shape['label']
                    label_counts[label] += 1

    return dict(label_counts)

def export_counts_to_excel(label_counts, excel_file):
    """
    Exports the label counts to an Excel file.

    Parameters:
    label_counts (dict): Dictionary containing label counts.
    excel_file (str): Path to the output Excel file.
    """
    df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    df.to_excel(excel_file, index=False)

class AnnotationHandler(FileSystemEventHandler):
    def __init__(self, annotation_dir, excel_path):
        self.annotation_dir = annotation_dir
        self.excel_path = excel_path
        self.update_class_counts()

    def update_class_counts(self):
        label_counts = count_objects_in_json_format(self.annotation_dir)
        export_counts_to_excel(label_counts, self.excel_path)

    def on_modified(self, event):
        if event.src_path.endswith(".json"):
            self.update_class_counts()

    def on_created(self, event):
        if event.src_path.endswith(".json"):
            self.update_class_counts()

    def on_deleted(self, event):
        if event.src_path.endswith(".json"):
            self.update_class_counts()

if __name__ == "__main__":
    annotation_directory = "path"
    excel_file = 'object_counts.xlsx'

    event_handler = AnnotationHandler(annotation_directory, excel_file)
    observer = Observer()
    observer.schedule(event_handler, path=annotation_directory, recursive=False)
    observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    print(f"Object counts are being monitored and updated in {excel_file}")
