
import os
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

    def get_payload(self, index):
        """
        Get payload data at `index`
        """
        payload = {}
        for field in self.fields:
            start = self.index_files[field][index]
            end   = self.index_files[field][index + 1]

            payload[field] = self.data_mmaps[field][start:end].tobytes().decode(self.encoding)
        
        return payload