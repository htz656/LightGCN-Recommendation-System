import torch


def sparse_dropout(sparse_mat: torch.Tensor, dropout_rate: float):
    nnz = sparse_mat._nnz()
    noise = torch.rand(nnz, device=sparse_mat.device)
    keep_mask = noise >= dropout_rate
    indices = sparse_mat._indices()[:, keep_mask]
    values = sparse_mat._values()[keep_mask] * (1.0 / (1 - dropout_rate))
    return torch.sparse_coo_tensor(indices, values, sparse_mat.shape, device=sparse_mat.device).coalesce()
