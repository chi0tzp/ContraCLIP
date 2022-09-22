import sys
import numpy as np
import mpmath
import torch
import torch.nn as nn


def norm(v, p=2, dim=0, eps=1e-12):
    """

    Args:
        v ():
        p (int):
        dim (int):
        eps (float):

    Returns:

    """
    return v.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(v)


class vMFLogPartition(torch.autograd.Function):
    """Evaluates log C_d(kappa) for vMF density -- Allows autograd wrt kappa."""

    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):
        """

        Args:
            ctx ()   :
            *args () :

            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape

        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa

        """
        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)

        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):

        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s + 1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI)

        if (logI2 != logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa


class vMF(nn.Module):
    """Calculate von Mises-Fisher density vMF(x; mu, kappa)."""
    def __init__(self, x_dim, reg=1e-6):
        super(vMF, self).__init__()
        self.x_dim = x_dim
        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.logkappa = nn.Parameter(0.01 * torch.randn([]))
        self.reg = reg
        self.realmin = 1e-10

    def set_params(self, mu, kappa):
        with torch.no_grad():
            self.mu_unnorm.copy_(mu)
            self.logkappa.copy_(torch.log(kappa + self.realmin))

    def get_params(self):
        mu = self.mu_unnorm / norm(self.mu_unnorm)
        kappa = self.logkappa.exp() + self.reg

        return mu, kappa

    def forward(self, x, utc=False):
        """Evaluate logliks, log p(x)

        Args:
            x ()   : batch of x's
            utc () : whether to evaluate only up to constant or exactly:
                        -- if True, no log-partition computed
                        -- if False, exact loglik computed

        Returns:
            logliks = log p(x)

        """
        mu, kappa = self.get_params()
        dotp = (mu.unsqueeze(0) * x).sum(1)

        if utc:
            logliks = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)
            logliks = kappa * dotp + logC

        return logliks

    def sample(self, N=1, rsf=10):
        """

        Args:
            N ()   : number of samples to generate
            rsf () : multiplicative factor for extra backup samples in rejection sampling

        Returns:
            samples; N samples generated

        Notes:
            no autodiff
        """

        d = self.x_dim
        with torch.no_grad():

            mu, kappa = self.get_params()

            # Step-1: Sample uniform unit vectors in R^{d-1}
            v = torch.randn(N, d - 1).to(mu)
            v = v / norm(v, dim=1)

            # Step-2: Sample v0
            kmr = np.sqrt(4 * kappa.item() ** 2 + (d - 1) ** 2)
            bb = (kmr - 2 * kappa) / (d - 1)
            aa = (kmr + 2 * kappa + d - 1) / 4
            dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
            beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))
            uniform = torch.distributions.Uniform(0.0, 1.0)
            v0 = torch.tensor([]).to(mu)
            while len(v0) < N:
                eps = beta.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                uns = uniform.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
                t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
                det = (d - 1) * t0.log() - t0 + dd - uns.log()
                v0 = torch.cat([v0, torch.tensor(w0[det >= 0]).to(mu)])
                if len(v0) > N:
                    v0 = v0[:N]
                    break
            v0 = v0.reshape([N, 1])

            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples = torch.cat([v0, (1 - v0 ** 2).sqrt() * v], 1)

            # Setup-4: Householder transformation
            e1mu = torch.zeros(d, 1).to(mu)
            e1mu[0, 0] = 1.0
            e1mu = e1mu - mu if len(mu.shape) == 2 else e1mu - mu.unsqueeze(1)
            e1mu = e1mu / norm(e1mu, dim=0)
            samples = samples - 2 * (samples @ e1mu) @ e1mu.t()

        return samples


class MixvMF(nn.Module):
    """Mixture of von Mises-Fisher (vMF): MixvMF(x) = \\sum_{m=1}^M \alpha_m vMF(x; mu_m, kappa_m)."""
    def __init__(self, x_dim, order, reg=1e-6):
        super(MixvMF, self).__init__()
        self.x_dim = x_dim
        self.order = order
        self.reg = reg
        self.alpha_logit = nn.Parameter(0.01 * torch.randn(order))
        self.comps = nn.ModuleList(
            [vMF(x_dim, reg) for _ in range(order)]
        )
        self.realmin = 1e-10

    def set_params(self, alpha, mus, kappas):

        with torch.no_grad():
            self.alpha_logit.copy_(torch.log(alpha + self.realmin))
            for m in range(self.order):
                self.comps[m].mu_unnorm.copy_(mus[m])
                self.comps[m].logkappa.copy_(torch.log(kappas[m] + self.realmin))

    def get_params(self):

        logalpha = self.alpha_logit.log_softmax(0)

        mus, kappas = [], []
        for m in range(self.order):
            mu, kappa = self.comps[m].get_params()
            mus.append(mu)
            kappas.append(kappa)

        # REVIEW
        # mus = torch.stack(mus, axis=0)
        # kappas = torch.stack(kappas, axis=0)
        mus = torch.stack(mus, dim=0)
        kappas = torch.stack(kappas, dim=0)

        return logalpha, mus, kappas

    def forward(self, x):
        """Evaluate logliks, log p(x)

        Args:
            x (): batch of x

        Returns:
            logliks = log p(x)
            logpcs = log p(x|c=m)
        """
        logalpha = self.alpha_logit.log_softmax(0)

        logpcs = []
        for m in range(self.order):
            logpcs.append(self.comps[m](x))
        logpcs = torch.stack(logpcs, dim=1)

        logliks = (logalpha.unsqueeze(0) + logpcs).logsumexp(1)

        return logliks, logpcs

    def sample(self, N=1, rsf=10):
        """

        Args:
            N ()   : number of samples to generate
            rsf () : multiplicative factor for extra backup samples in rejection sampling (used in sampling from vMF)

        Returns:
            samples = N samples generated
            cids = which components the samples come from; N-dim {0,1,...,M-1}-valued

        Notes:
            no autodiff
        """

        with torch.no_grad():
            alpha = self.alpha_logit.log_softmax(0).exp()
            cids = torch.multinomial(alpha, N, replacement=True)
            samples = torch.zeros(N, self.x_dim)
            for c in range(self.order):
                Nc = (cids == c).sum()
                if Nc > 0:
                    samples[cids == c, :] = self.comps[c].sample(N=Nc, rsf=rsf)

        return samples, cids


class MixvMFModel:
    def __init__(self, dim, order, em_max_iters=100, rll_tol=1e-6, verbose=False):
        self.dim = dim
        self.order = order
        self.em_max_iters = em_max_iters
        self.rll_tol = rll_tol
        self.realmin = 1e-10
        self.verbose = verbose

        # TODO: add comment
        self.mix = MixvMF(x_dim=self.dim, order=self.order)

    def get_params(self):
        return self.mix.get_params()

    def fit_em(self, X):
        samples = X
        ########################################################################################################
        ##                                                                                                    ##
        ##                                          [ EM-algorithm ]                                          ##
        ##                                                                                                    ##
        ########################################################################################################
        ll_old = -np.inf
        with torch.no_grad():
            for steps in range(self.em_max_iters):

                ########################################################################################################
                ##                                                                                                    ##
                ##                                             [ E-step ]                                             ##
                ##                                                                                                    ##
                ########################################################################################################
                logalpha, mus, kappas = self.mix.get_params()
                logliks, logpcs = self.mix(samples)
                ll = logliks.sum()
                jll = logalpha.unsqueeze(0) + logpcs
                qz = jll.log_softmax(1).exp()

                if steps == 0:
                    prn_str = '[Before EM starts] loglik = %.4f\n' % ll.item()
                else:
                    prn_str = '[Steps %03d] loglik (before M-step) = %.4f\n' % (steps, ll.item())
                print(prn_str)

                # tolerance check
                if steps > 0:
                    rll = (ll - ll_old).abs() / (ll_old.abs() + self.realmin)
                    if rll < self.rll_tol:
                        prn_str = 'Stop EM since the relative improvement '
                        prn_str += '(%.6f) < tolerance (%.6f)\n' % (rll.item(), self.rll_tol)
                        print(prn_str)
                        break

                ll_old = ll

                ########################################################################################################
                ##                                                                                                    ##
                ##                                             [ M-step ]                                             ##
                ##                                                                                                    ##
                ########################################################################################################
                qzx = (qz.unsqueeze(2) * samples.unsqueeze(1)).sum(0)
                qzx_norms = norm(qzx, dim=1)
                mus_new = qzx / qzx_norms
                Rs = qzx_norms[:, 0] / (qz.sum(0) + self.realmin)
                kappas_new = (self.mix.x_dim * Rs - Rs ** 3) / (1 - Rs ** 2)
                alpha_new = qz.sum(0) / samples.shape[0]

                # assign new params
                self.mix.set_params(alpha_new, mus_new, kappas_new)

            logliks, logpcs = self.mix(samples)
            ll = logliks.sum()
            prn_str = '[Training done] loglik = %.4f\n' % ll.item()
            print(prn_str)

    def sgd(self):
        raise NotImplementedError


class MixvMFGrad(nn.Module):
    def __init__(self, alphas, mus, kappas):
        super(MixvMFGrad, self).__init__()
        self.alphas = alphas
        self.mus = mus
        self.kappas = kappas

    def get_logC(self):
        # Calculate logC
        d = self.mus.shape[1]
        kappa = self.kappas
        ss = 0.5 * d - 1
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(ss, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)
        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        logC = d * vMFLogPartition.nhlog2pi + ss * kappa.log() - logI

        return logC


    @staticmethod
    def orthogonal_projection(s, w):
        """Orthogonally project the (n+1)-dimensional vector w onto the tangent space T_sS^n.

        Args:
            s (torch.Tensor): point on S^n
            w (torch.Tensor): (n+1)-dimensional vector to be projected on T_sS^n

        Returns:
            Pi_s(w) (torch.Tensor): orthogonal projection of w onto T_sS^n

        """
        # Get batch size (bs) and dimensionality of the ambient space (dim=n+1)
        bs, dim = s.shape

        # Calculate orthogonal projection
        I_ = torch.eye(dim).reshape(1, dim, dim).repeat(bs, 1, 1)
        X = I_ - torch.matmul(s.unsqueeze(2), s.unsqueeze(1))

        return torch.matmul(w.unsqueeze(1), X).squeeze(1)

    def forward(self, s):
        """

        Args:
            s (torch.Tensor): batch of points

        Returns:

        """
        # Calculate logC
        d = s.shape[1]
        kappa = self.kappas
        ss = 0.5 * d - 1
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(ss, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)
        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')
        logC = d * vMFLogPartition.nhlog2pi + ss * kappa.log() - logI


        #

        d = torch.einsum('ik, jk -> ij', s, self.mus)
        kd = self.kappas * d

        log_akcexp = torch.log(self.alphas) + torch.log(self.kappas) + logC + kd
        akcexp = torch.exp(log_akcexp)
        akcexp_mus = torch.einsum('ik, kj -> ij', akcexp, self.mus)
        gradients = akcexp_mus

        # ak = self.alphas * self.kappas * torch.exp(logC)
        # log_akc_exp_kd = torch.log(ak) + logC + kd
        # ak_exp_kd = torch.exp(log_akc_exp_kd)
        # ak_exp_kd_mus = torch.einsum('ik, kj -> ij', ak_exp_kd, self.mus)
        # gradients = ak_exp_kd_mus

        gradients = gradients / torch.norm(gradients, dim=1, keepdim=True)

        print("gradients")
        print(gradients)

        gradients = self.orthogonal_projection(s=s, w=gradients)

        return gradients


# === TODO: +++
class vMFGradient(nn.Module):
    def __init__(self, params_dict):
        """

        Args:
            params_dict (dict): TODO: +++
        """
        super(vMFGradient, self).__init__()
        # === Get vMF alphas ===
        if 'alphas' not in params_dict:
            raise KeyError("`alphas` not found in vMF model dictionary.")
        else:
            self.alphas = nn.Parameter(data=torch.from_numpy(np.array(params_dict['alphas'])).float(),
                                       requires_grad=False)

        # === Get vMF kappas ===
        if 'kappas' not in params_dict:
            raise KeyError("`kappas` not found in vMF model dictionary.")
        else:
            self.kappas = nn.Parameter(data=torch.from_numpy(np.array(params_dict['kappas'])).float(),
                                       requires_grad=False)

        # === Get vMF mus ===
        if 'mus' not in params_dict:
            raise KeyError("`mus` not found in vMF model dictionary.")
        else:
            self.mus = nn.Parameter(data=torch.from_numpy(np.array(params_dict['mus'])).float(),
                                    requires_grad=False)

        # === Get vMF logC ===
        if 'logC' not in params_dict:
            raise KeyError("`logC` not found in vMF model dictionary.")
        else:
            self.logC = nn.Parameter(data=torch.from_numpy(np.array(params_dict['logC'])).float(),
                                     requires_grad=False)

    @staticmethod
    def orthogonal_projection(s, w):
        """Orthogonally project the (n+1)-dimensional vector w onto the tangent space T_sS^n.

        Args:
            s (torch.Tensor): point on S^n
            w (torch.Tensor): (n+1)-dimensional vector to be projected on T_sS^n

        Returns:
            Pi_s(w) (torch.Tensor): orthogonal projection of w onto T_sS^n

        """
        # Get batch size (bs) and dimensionality of the ambient space (dim=n+1)
        bs, dim = s.shape

        # Calculate orthogonal projection
        I_ = torch.eye(dim).reshape(1, dim, dim).repeat(bs, 1, 1)
        X = I_ - torch.matmul(s.unsqueeze(2), s.unsqueeze(1))

        return torch.matmul(w.unsqueeze(1), X).squeeze(1)

    def forward(self, s):
        """

        Args:
            s (torch.Tensor):

        Returns:

        """

        d = torch.einsum('ik, jk -> ij', s, self.mus)
        kd = self.kappas * d

        log_akcexp = torch.log(self.alphas) + torch.log(self.kappas) + self.logC + kd

        print("log_akcexp")
        print(log_akcexp.shape)
        sys.exit()


        akcexp = torch.exp(log_akcexp)
        akcexp_mus = torch.einsum('ik, kj -> ij', akcexp, self.mus)
        gradients = akcexp_mus

        # kd = self.kappas * d
        # log_akc_exp_kd = torch.log(ak) + self.logC + kd
        # ak_exp_kd = torch.exp(log_akc_exp_kd)
        # ak_exp_kd_mus = torch.einsum('ik, kj -> ij', ak_exp_kd, self.mus)
        #
        # gradients = ak_exp_kd_mus
        # gradients = gradients / torch.norm(gradients, dim=1, keepdim=True)
        #
        # print("logC")
        # print(self.logC)
        #
        # print("gradients")
        # print(torch.norm(gradients, dim=1))
        # sys.exit()

        gradients = self.orthogonal_projection(s=s, w=gradients)

        return gradients

