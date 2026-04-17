import torch
import torch.nn.functional as F

def curvature_tv_prox(
    depth,
    depth_ref,
    lambda_curv=0.1,
    rho=1.0,
    iters=10,
    tau=0.125,
):
    """
    Solve:
        min_z lambda * TV(z) + rho/2 ||z - depth_ref||^2
    via simple gradient descent (can be replaced by PD later)
    """
    z = depth.clone().detach().requires_grad_(True)

    for _ in range(iters):
        dzx = z[:, :, 1:, :] - z[:, :, :-1, :]
        dzy = z[:, :, :, 1:] - z[:, :, :, :-1]
        tv = (dzx.abs().mean() + dzy.abs().mean())

        fidelity = 0.5 * rho * ((z - depth_ref) ** 2).mean()
        loss = lambda_curv * tv + fidelity
        loss.backward()

        with torch.no_grad():
            z -= tau * z.grad
            z.grad.zero_()

    return z.detach()
