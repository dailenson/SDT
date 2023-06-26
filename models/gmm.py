import torch

### split final output of our model into Mixture Density Network (MDN) parameters and pen state
def get_mixture_coef(output):
    z = output
    z_pen_logits = z[:, 0:3]  # pen state

    # MDN parameters are used to predict the pen moving
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z[:, 3:], 20, 1) 

    # softmax pi weights:
    z_pi = torch.softmax(z_pi, -1)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = torch.minimum(torch.exp(z_sigma1), torch.Tensor([500.0]).cuda())
    z_sigma2 = torch.minimum(torch.exp(z_sigma2), torch.Tensor([500.0]).cuda())
    z_corr = torch.tanh(z_corr)
    result = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits]
    return result

### generate the pen moving and state from the predict output
def get_seq_from_gmm(gmm_pred):
    gmm_pred = gmm_pred.reshape(-1, 123)
    [pi, mu1, mu2, sigma1, sigma2, corr, pen_logits] = get_mixture_coef(gmm_pred)
    max_mixture_idx = torch.stack([torch.arange(pi.shape[0], dtype=torch.int64).cuda(), torch.argmax(pi, 1)], 1)
    next_x1 = mu1[list(max_mixture_idx.T)]
    next_x2 = mu2[list(max_mixture_idx.T)]
    pen_state = torch.argmax(gmm_pred[:, :3], dim=-1)
    pen_state = torch.nn.functional.one_hot(pen_state, num_classes=3).to(gmm_pred)
    seq_pred = torch.cat([next_x1.unsqueeze(1), next_x2.unsqueeze(1), pen_state],-1)
    return seq_pred