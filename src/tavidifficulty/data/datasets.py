import numpy as np
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, mrt_slice_data, y):
        assert len(mrt_slice_data) == len(y)
        self.mrt_slice_data = mrt_slice_data
        self.y = y

    def __len__(self):
        return len(self.mrt_slice_data)

    def __getitem__(self, index):
        current_mrt_slice_data = self.mrt_slice_data[index]
        current_y = self.y[index]

        # get into proper shape
        current_mrt_slice_data = current_mrt_slice_data.type(torch.float32)
        current_y = current_y.type(torch.float32)

        return current_mrt_slice_data, current_y