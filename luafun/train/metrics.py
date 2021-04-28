import json
import uuid
import os
import io

import torch


class ReservedName(RuntimeError):
    pass


class MetricWriter:
    def __init__(self, folder, uid=None):
        self.folder = folder
        if uid is None:
            uid = uuid.uuid4().hex
        self.uid = uid
        os.makedirs(os.path.join(self.folder, self.uid), exist_ok=True)

    def open(self, file, mode):
        return open(os.path.join(self.folder, self.uid, file), mode)

    def save_metrics(self, metrics):
        with self.open('metrics.ajson', mode='a') as file:
            jsonstr = f'{json.dumps(metrics)}\n'
            file.write(jsonstr)

    def load_metrics(self):
        with self.open('metrics.ajson', mode='r') as file:
            lines = file.read().split('\n')
            metrics = []

            for line in lines:
                if line == '':
                    continue

                metrics.append(json.loads(line))

            return metrics

    def save_meta(self, metadata):
        with self.open('meta.json', 'w') as meta:
            meta.write(json.dumps(metadata))

    def load_meta(self):
        with self.open('meta.json', 'r') as meta:
            return json.loads(meta.read())

    def save_weights(self, model, weights):
        with self.open(f'{model}.dat', 'wb') as meta:
            meta.write(weights)

    def load_weights(self, model):
        with self.open(f'{model}.dat', 'rb') as meta:
            return torch.load(io.BytesIO(meta.read()))
