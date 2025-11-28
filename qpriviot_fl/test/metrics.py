import json
import os
from datetime import datetime
from typing import Dict, List

class MetricsLogger:
    def __init__(self):
        self.history: List[Dict] = []
        self.start_time = datetime.now().isoformat()

    def log_round(self, metrics: Dict):
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'): clean_metrics[k] = v.item()
            else: clean_metrics[k] = v
        self.history.append(clean_metrics)

    def save_to_file(self, filename: str):
        data = {"start_time": self.start_time, "rounds": self.history}
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)