import torch


class Collator(object):
    def __init__(self, pimgr):
        self.pimgr = pimgr

    def __call__(self, dataset_items):
        all_labels = torch.tensor([item['label'] for item in dataset_items])

        all_items = [item['item'] for item in dataset_items]
        batch_items = torch.stack(all_items)

        all_pds = [item['pd'] for item in dataset_items]

        lengths = [len(pd) for pd in all_pds]
        max_pd_len = max(lengths)
        pd_feature_size = all_pds[0].shape[-1]
        lengths = torch.tensor(lengths)
        mask = torch.arange(max_pd_len)[None, :] < lengths[:, None]

        batch_pd = torch.zeros(len(dataset_items), max_pd_len, pd_feature_size)
        for i, pd in enumerate(all_pds):
            batch_pd[i][:len(pd)] = pd

        if self.pimgr is not None:
            PI = torch.from_numpy(self.pimgr.fit_transform(batch_pd)).to(torch.float32)
            max_val = PI.max(dim=1, keepdim=True)[0]
            PI = torch.where(max_val > 1e-10, PI / PI.max(dim=1, keepdim=True)[0], PI)
        else:
            PI = None

        return {
            'labels': all_labels,
            'pds': batch_pd,
            'items': batch_items,
            'lengths': lengths,
            'mask': mask,
            'pis': PI
        }
