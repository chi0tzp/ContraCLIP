import torch
from torch import nn


class CorpusSupportSets(nn.Module):
    def __init__(self, semantic_dipoles_features, gamma=1.0, learn_gammas=False):
        """CorpusSupportSets class constructor.

        Args:
            semantic_dipoles_features (torch.Tensor) : CLIP text feature statistics of prompts from the given corpus
            gamma (float)                            : initial gamma parameters of the RBFs in the Vision-Language space
            learn_gammas (bool)                      : TODO: +++
        """
        super(CorpusSupportSets, self).__init__()
        self.semantic_dipoles_features = semantic_dipoles_features
        self.gamma = gamma
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
        # Define RBF loggammas
        self.LOGGAMMA = nn.Parameter(data=torch.log(torch.scalar_tensor(self.gamma)) * torch.ones(self.num_support_sets, 2),
                                     requires_grad=self.learn_gammas)

        # self.LOGGAMMA = nn.Parameter(data=torch.ones(self.num_support_sets, 2), requires_grad=self.learn_gammas)
        # for k in range(self.num_support_sets):
        #     gammas = -torch.log(torch.Tensor([self.beta, self.beta])) / \
        #         (self.semantic_dipoles_features[k, 1] - self.semantic_dipoles_features[k, 0]).norm() ** 2
        #     self.LOGGAMMA.data[k] = torch.log(gammas)

    @staticmethod
    def orthogonal_projection(s, w):
        """Orthogonally project the (n+1)-dimensional vector w onto the tangent space T_sS^n.

        Args:
            s (torch.Tensor): point on S^n
            w (torch.Tensor): (n+1)-dimensional vector to be projected on T_sS^n

        Returns:
            P_s(w) (torch.Tensor): orthogonal projection of w onto T_sS^n

        """
        # Get batch size (bs) and dimensionality of the ambient space (dim=n+1)
        bs, dim = s.shape

        # REVIEW:
        I_ = torch.eye(dim).reshape(1, dim, dim).repeat(bs, 1, 1)
        X = I_ - torch.matmul(s.unsqueeze(2), s.unsqueeze(1))

        return torch.matmul(w.unsqueeze(1), X).squeeze(1)

    def forward(self, support_sets_mask, z):
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2, 1) - support_sets_batch

        grad_f = -2 * (alphas_batch * gammas_batch *
                       torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        # TODO: add comment
        grad_f = self.orthogonal_projection(s=z, w=grad_f)

        # return grad_f
        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
