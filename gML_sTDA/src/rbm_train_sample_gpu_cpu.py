import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob, torch.bernoulli(prob)

    def free_energy(self, v):
        # v: (batch, n_visible)
        vbias_term = v.mv(self.v_bias)  # (batch,)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term


def filter_single_excitation(data, n_occ_keep, n_vir_keep):
    valid_mask = np.array([
        (np.sum(v[:n_occ_keep] == 0) == 1) and
        (np.sum(v[n_occ_keep:] == 1) == 1)
        for v in data
    ])
    return data[valid_mask]


def project_to_single_excitation(v, n_occ_keep, n_vir_keep):
    v = v.copy()
    for i in range(v.shape[0]):
        occ = v[i, :n_occ_keep]
        vir = v[i, n_occ_keep:]

        if np.sum(occ == 0) != 1:
            occ[:] = 1
            occ[np.random.randint(0, n_occ_keep)] = 0
        if np.sum(vir == 1) != 1:
            vir[:] = 0
            vir[np.random.randint(0, n_vir_keep)] = 1

        v[i, :n_occ_keep] = occ
        v[i, n_occ_keep:] = vir
    return v


def _resolve_device(use_gpu: bool | None = None, device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if use_gpu:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')


def train_rbm(rbm, data, lr=0.01, batch_size=32, epochs=200, use_gpu: bool = True, device=None):

    dev = _resolve_device(use_gpu=use_gpu, device=device)

    rbm = rbm.to(dev)
    data = torch.tensor(data, dtype=torch.float32, device=dev)

    n_samples = data.size(0)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=dev)
        epoch_loss = 0.0

        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i + batch_size]]

            prob_h, h = rbm.sample_h(batch)
            prob_v, v_recon = rbm.sample_v(h)
            prob_h_neg, _ = rbm.sample_h(v_recon)

            positive_grad = torch.matmul(prob_h.t(), batch)
            negative_grad = torch.matmul(prob_h_neg.t(), v_recon)

            rbm.W.grad = -(positive_grad - negative_grad) / batch.size(0)
            rbm.v_bias.grad = -(torch.mean(batch - v_recon, dim=0))
            rbm.h_bias.grad = -(torch.mean(prob_h - prob_h_neg, dim=0))

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss = torch.mean(rbm.free_energy(batch)) - torch.mean(rbm.free_energy(v_recon))
            epoch_loss += loss.item()

    return rbm


@torch.no_grad()
def sample_from_rbm_constrained(rbm, n_samples, n_occ_keep, n_vir_keep, k_steps=50, device=None, force_cpu_output=True):

    dev = _resolve_device(device=device, use_gpu=None)
    if device is None:
        dev = next(rbm.parameters()).device

    v0 = torch.bernoulli(torch.rand(n_samples, rbm.W.size(1))).cpu().numpy()
    v0 = project_to_single_excitation(v0, n_occ_keep, n_vir_keep).astype(np.float32)
    v = torch.tensor(v0, dtype=torch.float32, device=dev)

    for _ in range(k_steps):
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

        v_np = v.detach().cpu().numpy()
        v_np = project_to_single_excitation(v_np, n_occ_keep, n_vir_keep).astype(np.float32)
        v = torch.tensor(v_np, dtype=torch.float32, device=dev)

    if force_cpu_output:
        return v.detach().cpu().numpy().astype(int)
    return v
