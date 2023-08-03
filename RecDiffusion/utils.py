import torch

def tensor_to_dict(pose: torch.tensor, node_attr=None):
    assert pose.shape[-1] == 3
    data = {'pos': pose}
    if node_attr:
        data['z'] = node_attr
    # data['pos'] = pose
    return data