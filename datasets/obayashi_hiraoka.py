from datasets.base_dataset import BasePDDataset
import torch
from torchvision import transforms


class ObayashiHiraoka(BasePDDataset):
    def __getitem__(self, idx):
        item, label = self.dataset[idx]
        norm = transforms.Normalize((0.5,), (0.5,))
        item = norm(item)
        pd = torch.from_numpy(self.pds[idx])
        pd = pd[pd[:, 2] == 1]
        pd = pd[:, :2]

        if self.leave is not None:
            if len(pd) >= self.leave:
                lifetime = pd[:, 1] - pd[:, 0]
                order = torch.argsort(lifetime, descending=True)
                pd = pd[order][:self.leave]

        return {
            'item': item,
            'pd': pd,
            'label': label
        }
