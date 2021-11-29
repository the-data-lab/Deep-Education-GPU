import torch as th
import torch.utils.dlpack
import kernel as gpk

def gp_gspmm(g, X, dim0, dim1, inverse, norm):
    X_dl = th.utils.dlpack.to_dlpack(X)

    # declare the output tensor here
    cuda0 = th.device('cuda')
    res = th.zeros(dim0, dim1, device=cuda0)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmm(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

    return res
