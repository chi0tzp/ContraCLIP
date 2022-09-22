import torch
from torch import nn
import numpy as np


class CorpusSupportSets(nn.Module):
    def __init__(self, semantic_dipoles_features, semantic_dipoles_covariances, gammas, gamma_0=1.0,
                 learn_gammas=False):
        """CorpusSupportSets class constructor.

        Args:
            semantic_dipoles_features (torch.Tensor)    : TODO: +++
            semantic_dipoles_covariances (torch.Tensor) : TODO: +++
            gammas (str)                                : TODO: +++
            gamma_0 (float)                             : TODO: +++
            learn_gammas (bool)                         : TODO: +++

        """
        super(CorpusSupportSets, self).__init__()
        self.semantic_dipoles_features = semantic_dipoles_features
        self.semantic_dipoles_covariances = semantic_dipoles_covariances
        self.gammas = gammas
        self.gamma_0 = gamma_0
        self.learn_gammas = learn_gammas

        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ Corpus Support Sets (CSS) ]                                       ##
        ##                                                                                                            ##
        ################################################################################################################
        # Initialization
        self.num_support_sets = self.semantic_dipoles_features.shape[0]
        self.support_vectors_dim = self.semantic_dipoles_features.shape[2]

        ############################################################################################################
        ##                                      [ SUPPORT_SETS: (K, 2, d) ]                                       ##
        ############################################################################################################
        self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets, 2 * self.support_vectors_dim),
                                         requires_grad=False)
        self.SUPPORT_SETS.data = self.semantic_dipoles_features.reshape(self.semantic_dipoles_features.shape[0],
                                                                        self.semantic_dipoles_features.shape[1] *
                                                                        self.semantic_dipoles_features.shape[2]).clone()

        ############################################################################################################
        ##                                          [ ALPHAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Define alphas as pairs of [-1, 1] for each dipole
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2), requires_grad=False)
        for k in range(self.num_support_sets):
            self.ALPHAS.data[k] = torch.Tensor([1, -1])

        ############################################################################################################
        ##                                          [ GAMMAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Spherical gammas
        if self.gammas == 'spherical':
            self.LOGGAMMA = nn.Parameter(data=np.log(self.gamma_0) * torch.ones(self.num_support_sets, 2),
                                         requires_grad=self.learn_gammas)
        # Diagonal gammas
        elif self.gammas == 'diag':
            # self.LOGGAMMA = nn.Parameter(
            #     data=torch.log(torch.div(self.gamma_0, self.semantic_dipoles_covariances.reshape(-1, 2 * self.support_vectors_dim))),
            #     requires_grad=self.learn_gammas)

            # TODO: add comment
            # semantic_dipoles_covariances = 1.0 / self.semantic_dipoles_covariances
            semantic_dipoles_covariances = torch.div(1.0, self.semantic_dipoles_covariances)
            semantic_dipoles_covariances = semantic_dipoles_covariances.reshape(-1, 2 * self.support_vectors_dim)
            self.LOGGAMMA = nn.Parameter(data=torch.log(self.gamma_0 * semantic_dipoles_covariances),
                                         requires_grad=self.learn_gammas)

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

    @staticmethod
    def exponential_map(s, u):
        """Calculate the exponential map from the tangent space T_sS^n to the sphere S^n.

        Args:
            s (torch.Tensor): point on T_sS^n
            u (torch.Tensor): exponential mapping of s onto S^n

        Returns:
            exp_s(u) (torch.Tensor): exponential map of u in T_sS^n onto S^n.
        """

        u_norm = torch.norm(u, dim=1, keepdim=True)
        cos_u = torch.cos(u_norm)
        sin_u = torch.sin(u_norm)

        return cos_u * s + sin_u * u / u_norm

    def logarithmic_map(self, s, q):
        """Calculate the logarithmic map of a sphere point q onto the tangent space TsS^n.

        Args:
            s (torch.Tensor): point on S^n defining the tangent space TsS^n
            q (torch.Tensor): point on S^n

        Returns:
            log_s(q) (torch.Tensor): logarithmic map of q onto the tangent space TsS^n.

        """
        pi_s_q_minus_s = self.orthogonal_projection(s, q-s)

        return torch.arccos((q * s).sum(axis=-1)).unsqueeze(1) * pi_s_q_minus_s / \
            torch.norm(pi_s_q_minus_s, dim=1, keepdim=True)

    def forward(self, support_sets_mask, z):

        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        grad_f = None
        if self.gammas == 'spherical':
            # Get batch of RBF gamma/log(gamma) parameters
            gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))

            # Calculate grad of f at z
            D = z.unsqueeze(dim=1) - support_sets_batch
            grad_f = -2 * (alphas_batch * gammas_batch *
                           torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        elif self.gammas == 'diag':
            # Get batch of RBF gamma/log(gamma) parameters
            gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA)).reshape(
                -1, 2, self.support_vectors_dim)

            # Calculate grad of f at z
            D = z.unsqueeze(dim=1) - support_sets_batch
            SGSt = torch.einsum('b i d, b i d -> b i', D ** 2, gammas_batch)
            SG = torch.einsum('b i d, b i d -> b i d', D, gammas_batch)
            grad_f = -(alphas_batch * torch.exp(-SGSt.unsqueeze(dim=2)) * SG).sum(dim=1)

        # Orthogonally project gradient to the tangent space of z (Riemannian gradient)
        grad_f = self.orthogonal_projection(s=z, w=grad_f)

        return grad_f
