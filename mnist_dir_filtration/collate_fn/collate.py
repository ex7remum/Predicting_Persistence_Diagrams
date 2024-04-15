import torch

def collate_fn(dataset_items):
    all_labels = torch.tensor([item['label'] for item in dataset_items])
    
    all_items = [item['item'] for item in dataset_items]
    batch_items = torch.stack(all_items)
    
    all_pds = [item['pd'] for item in dataset_items]
    
    max_pd_len = max(len(pd) for pd in all_pds)
    pd_feature_size = all_pds[0].shape(-1)
    
    batch_pd = torch.zeros(len(dataset_items), max_pd_len, pd_feature_size)
    for i, pd in enumerate(all_pds):
        batch_pd[i][:len(pd)] = pd
        
    return {
        'labels': all_labels,
        'pds': batch_items,
        'items': batch_items
    }