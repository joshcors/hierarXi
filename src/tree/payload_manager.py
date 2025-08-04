
import os
import json
import numpy as np

class PayloadManger:
    def __init__(self, fields, payload_dir, _id="arxiv", encoding="utf-8"):
        self.fields = fields
        self._id = _id
        self.encoding = encoding

        self.data_mmaps = {
            field : np.memmap(os.path.join(payload_dir, f"{_id}_{field}.bin"), mode="r", dtype=np.uint8) for field in self.fields
        }
        self.index_files = {
            field : np.fromfile(os.path.join(payload_dir, f"{_id}_{field}.idx"), dtype=np.uint64) for field in self.fields
        }
        self.data_path = None
        self.order_path = None
        self.info_path = None
        self.info = None
        self.order = None

    def load_info_and_order(self, data_dir):
        self.data_path = data_dir
        self.order_path = os.path.join(data_dir, "order.dat")
        self.info_path = os.path.join(data_dir, "info.json")
        
        with open(self.info_path, "r") as f:
            self.info = json.load(f)

        self.order = np.memmap(self.order_path, mode="readonly", shape=(self.info["cursor"], ), dtype=np.int32)

    def get_payload(self, index):
        """
        Get payload data at `index`
        """
        if self.order is not None:
            index = self.order[index]

        payload = {}
        for field in self.fields:
            start = self.index_files[field][index]
            end   = self.index_files[field][index + 1]

            payload[field] = self.data_mmaps[field][start:end].tobytes().decode(self.encoding)
        
        return payload