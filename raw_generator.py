from raw_type import raw_type


class generator(raw_type):
    def __init__(self):
        self.data = []
        self.label = []
        self.fs = 0
        self.lead = 6
        self.length = 0

    def read_data(self, path):
        import wfdb
        import os

        all_files = os.listdir(path)
        all_files = [i.split('.')[0] for i in all_files if 'hea' in i]
        from collections import Counter
        for i in all_files:
            file_path = os.path.join(path, i)
            if "102" in file_path or "104" in file_path or "107" in file_path or "217" in file_path:
                continue
            ann = wfdb.rdann(file_path, 'atr')
            if '114' not in file_path:
                data = wfdb.rdrecord(file_path).p_signal[:, 0]
            else:
                data = wfdb.rdrecord(file_path).p_signal[:, 0]

            self.data.append(data)
            self.label.append(ann)
            self.fs = ann.fs
            self.length = 650000


