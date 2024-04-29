from datasets.base_dataset import BasePDDataset
import torch


class OrbitDataset(BasePDDataset):
    def __getitem__(self, idx):
        item, label = self.dataset[idx]
        item = torch.from_numpy(item).to(torch.float32)
        pd = torch.from_numpy(self.pds[idx])
        pd = pd[pd[:, 2] == 1]

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
