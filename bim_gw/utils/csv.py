import pathlib

import csv
from copy import deepcopy

import numpy as np


def _set_float(val):
    try:
        return float(val)
    except ValueError as e:
        return val


class CSVLog:
    def __init__(self, file_path):
        self.index_to_keys = None
        self.keys_to_index = None
        self.data = []
        self.file_path = pathlib.Path(file_path)
        self.filters = []
        with open(self.file_path) as csvfile:
            reader = csv.reader(csvfile)
            for k, row in enumerate(reader):
                if k == 0:
                    self.index_to_keys = row
                    self.keys_to_index = {name: i for i, name in enumerate(row)}
                else:
                    row = self._get_dict_row(row)
                    # if row["metrics/val_supervision_loss (last)"] == '':
                    #     row["metrics/val_supervision_loss (last)"] = np.inf
                    if row["metrics/epoch (last)"] == '':
                        row["metrics/epoch (last)"] = '0'
                    row = {key: _set_float(val) for key, val in row.items()}
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def values(self, item):
        return [d[item] for d in self.data]

    def _get_dict_row(self, row):
        return {key: row[k] for key, k in self.keys_to_index.items()}

    def add_column(self, name, fn):
        new_log = deepcopy(self)
        new_data = []
        for row in self.data:
            row[name] = fn(row)
            new_data.append(row)
        new_log.data = new_data
        new_log.index_to_keys.append(name)
        new_log.keys_to_index[name] = new_log.index_to_keys.index(name)
        return new_log

    def add_token_column(self, name="_token"):
        return self.add_column(name, lambda row: int(row["Id"].split("-")[1]))

    def filter(self, cond):
        new_log = deepcopy(self)
        new_log.filters.append(cond)
        new_data = []
        for row in new_log.data:
            if cond(row):
                new_data.append(row)
        new_log.data = new_data
        return new_log

    def filter_eq(self, key, val):
        cond = lambda row: row[key] == val
        return self.filter(cond)

    def filter_neq(self, key, val):
        cond = lambda row: row[key] != val
        return self.filter(cond)

    def filter_between(self, key, mini=None, maxi=None):
        mini = mini if mini is not None else -np.inf
        maxi = maxi if maxi is not None else np.inf
        cond = lambda row: mini <= row[key] <= maxi
        return self.filter(cond)

    def filter_in(self, key, values):
        cond = lambda row: row[key] in values
        return self.filter(cond)

    def zip(self, keys, sort=False):
        outputs = [[] for k in range(len(keys))]
        for row in self.data:
            for k, key in enumerate(keys):
                outputs[k].append(row[key])
        if sort is not None and len(outputs[0]):
            outputs = zip(*sorted(zip(*outputs), key=sort))
        return outputs