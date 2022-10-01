import sys
import torch
from torch import nn


class CorpusSupportSets(nn.Module):
    def __init__(self, semantic_dipoles_cls, semantic_dipoles_means, semantic_dipoles_covariances, gamma_0=1.0):
        """CorpusSupportSets class constructor.

        Args:
            semantic_dipoles_cls (torch.Tensor)         : TODO: +++
            semantic_dipoles_means (torch.Tensor)       : TODO: +++
            semantic_dipoles_covariances (torch.Tensor) : TODO: +++
            gamma_0 (float)                             : TODO: +++

        """
        super(CorpusSupportSets, self).__init__()
        self.semantic_dipoles_cls = semantic_dipoles_cls
        self.semantic_dipoles_means = semantic_dipoles_means
        self.semantic_dipoles_covariances = semantic_dipoles_covariances
        self.gamma_0 = gamma_0

        # Initialization
        self.semantic_dipoles_features_cls = self.semantic_dipoles_cls
        self.num_support_sets = self.semantic_dipoles_features_cls.shape[0]
        self.support_vectors_dim = self.semantic_dipoles_features_cls.shape[2]

        ############################################################################################################
        ##                                                                                                        ##
        ##                                [ Semantic Dipoles' VL (Text) Features ]                                ##
        ##                                                                                                        ##
        ############################################################################################################

        ############################################################################################################
        ##                             [ SEMANTIC_DIPOLES_FEATURES_CLS: (K, 2 * d) ]                              ##
        ############################################################################################################
        self.SEMANTIC_DIPOLES_FEATURES_CLS = nn.Parameter(
            data=self.semantic_dipoles_features_cls.reshape(self.num_support_sets, 2 * self.support_vectors_dim),
            requires_grad=False)

        ############################################################################################################
        ##                                                                                                        ##
        ##                                          [ Warping Functions ]                                         ##
        ##                                                                                                        ##
        ############################################################################################################

        ############################################################################################################
        ##                                      [ SUPPORT_SETS: (K, 2, d) ]                                       ##
        ############################################################################################################
        # REVIEW: take the mean to CLS features
        # self.SUPPORT_SETS = nn.Parameter(
        #     data=self.semantic_dipoles_means.reshape(self.num_support_sets, 2 * self.support_vectors_dim),
        #     requires_grad=False)
        self.SUPPORT_SETS = nn.Parameter(
            data=self.semantic_dipoles_features_cls.reshape(self.num_support_sets, 2 * self.support_vectors_dim),
            requires_grad=False)

        ############################################################################################################
        ##                                          [ ALPHAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Define alphas as pairs of [-1, 1] for each dipole
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2), requires_grad=False)
        for k in range(self.num_support_sets):
            self.ALPHAS.data[k] = torch.Tensor([1, -1])

        ############################################################################################################
        ##                                        [ GAMMAS: (K, 2 * d) ]                                          ##
        ############################################################################################################
        # REVIEW: ================================================================================================ #
        # import matplotlib.pyplot as plt
        # kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
        # dipole_idx = 0
        # semantic_dipoles_covariances = torch.div(1, semantic_dipoles_covariances)
        # plt.hist(semantic_dipoles_covariances[dipole_idx, 0, :].cpu().detach().numpy(), **kwargs, color='g', label='Positive')
        # plt.hist(semantic_dipoles_covariances[dipole_idx, 1, :].cpu().detach().numpy(), **kwargs, color='r', label='Negative')
        # plt.show()

        print("self.gamma_0: {}".format(self.gamma_0))
        print(
            "*** self.gamma_0 * self.semantic_dipoles_covariances {} ***".format(self.semantic_dipoles_covariances.shape))
        print("\tmin  : {}".format((self.gamma_0 * self.semantic_dipoles_covariances).min()))
        print("\tmax  : {}".format((self.gamma_0 * self.semantic_dipoles_covariances).max()))
        print("\tmean : {}".format((self.gamma_0 * self.semantic_dipoles_covariances).mean()))
        print(
            "*** self.gamma_0 / self.semantic_dipoles_covariances {} ***".format(self.semantic_dipoles_covariances.shape))
        print("\tmin  : {}".format(torch.div(self.gamma_0, self.semantic_dipoles_covariances).min()))
        print("\tmax  : {}".format(torch.div(self.gamma_0, self.semantic_dipoles_covariances).max()))
        print("\tmean : {}".format(torch.div(self.gamma_0, self.semantic_dipoles_covariances).mean()))
        # sys.exit()
        # REVIEW: ================================================================================================ #

        # REVIEW: Use the variances as gammas
        # semantic_dipoles_covariances = self.gamma_0 * self.semantic_dipoles_covariances
        semantic_dipoles_covariances = torch.div(self.gamma_0, self.semantic_dipoles_covariances)

        print('*********************************')
        print(semantic_dipoles_covariances.min())
        print(semantic_dipoles_covariances.max())
        print(semantic_dipoles_covariances.mean())
        print('*********************************')

        sys.exit()

        self.LOGGAMMA = nn.Parameter(
            data=torch.log(semantic_dipoles_covariances.reshape(-1, 2 * self.support_vectors_dim)),
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
        # REVIEW: previous...
        # # Get RBF support sets batch
        # support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        # support_sets_batch = support_sets_batch.reshape(-1, 2, self.support_vectors_dim)
        #
        # # Get batch of RBF alpha parameters
        # alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)
        #
        # # REVIEW
        # alphas_batch = target_shift_signs.unsqueeze(1).unsqueeze(1) * alphas_batch
        #
        # # Get batch of RBF gamma/log(gamma) parameters
        # gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA)).reshape(
        #     -1, 2, self.support_vectors_dim)
        #
        # # Calculate grad of f at z
        # D = z.unsqueeze(dim=1) - support_sets_batch
        # SGSt = torch.einsum('b i d, b i d -> b i', D ** 2, gammas_batch)
        # SG = torch.einsum('b i d, b i d -> b i d', D, gammas_batch)
        # grad_f = -(alphas_batch * torch.exp(-0.5 * SGSt.unsqueeze(dim=2)) * SG).sum(dim=1)
        #
        # # Orthogonally project gradient to the tangent space of z (Riemannian gradient)
        # # grad_f = self.orthogonal_projection(s=z, w=grad_f)
        #
        # # grad_f = grad_f / torch.norm(grad_f, dim=1, keepdim=True)
        #
        # return grad_f

        # REVIEW: NEW
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2, self.support_vectors_dim)

        # Get batch of RBF gamma/log(gamma) parameters
        gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA)).reshape(
            -1, 2, self.support_vectors_dim)

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1) - support_sets_batch
        SGSt = torch.einsum('b i d, b i d -> b i', D ** 2, gammas_batch)
        SG = torch.einsum('b i d, b i d -> b i d', D, gammas_batch)
        grad_f = torch.exp(-0.5 * SGSt.unsqueeze(dim=2)) * SG

        return grad_f