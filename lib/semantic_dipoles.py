import sys
import torch
import clip


class SemanticDipoles:
    def __init__(self, corpus, clip_model, use_cuda, include_cls_in_mean=False):
        self.corpus = corpus
        self.use_cuda = use_cuda
        self.clip_model = clip_model.to('cuda' if self.use_cuda else 'cpu')
        self.include_cls_in_mean = include_cls_in_mean
        self.num_dipoles = len(self.corpus)
        self.dim = 512

        # Define CLIP text encoder for getting token representations
        self.transformer = self.clip_model.transformer
        self.positional_embedding = self.clip_model.positional_embedding
        self.ln_final = self.clip_model.ln_final
        self.text_projection = self.clip_model.text_projection

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

    def get_dipole_features(self):
        # Get CLIP text features for the given dipoles

        # TODO: rename `dipole_features` --> `dipole_features_cls`
        # TODO: add `dipole_features_tav`

        # Dipoles' features taken at the CLS (EOS / End of Sentence) token position
        dipole_features_cls = []

        # Dipoles' features averaged at all token positions -- REVIEW: all but CLS, or all tokens?
        dipole_features_token_mean = []

        # Dipoles' features' covariances calculated over all token positions -- REVIEW: all but CLS, or all tokens?
        dipole_features_token_cov = []

        for t in range(self.num_dipoles):
            # Get dipole's CLIP text representations
            dipole_features_at_cls = self.clip_model.encode_text(
                clip.tokenize(self.corpus[t]).to('cuda' if self.use_cuda else 'cpu'))

            # Normalize dipole's features
            dipole_features_at_cls /= torch.norm(dipole_features_at_cls, dim=1, keepdim=True)
            dipole_features_cls.append(dipole_features_at_cls.unsqueeze(0))

            # Get sample (token) covariances for the dipole
            tokenized_dipole = clip.tokenize(self.corpus[t])

            # Get CLS / EOS (EndOfSentence) token position
            cls_positions = tokenized_dipole.argmax(dim=-1)
            dipole_embeddings = self.clip_model.token_embedding(tokenized_dipole).type(self.clip_model.dtype)

            # Get dipole token representations
            x = dipole_embeddings + self.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.clip_model.dtype)

            # Include CLS token?
            stop_position = 0
            if not self.include_cls_in_mean:
                stop_position = 1

            positive_pole_token_representations = []
            negative_pole_token_representations = []
            for token_position in range(1, cls_positions.max() + 1):
                token_representations = x[torch.arange(x.shape[0]), token_position] @ self.text_projection
                if token_position <= cls_positions[0] - stop_position:
                    positive_pole_token_representations.append(token_representations[0].unsqueeze(0))
                if token_position <= cls_positions[1] - stop_position:
                    negative_pole_token_representations.append(token_representations[1].unsqueeze(0))
            positive_pole_token_representations = torch.cat(positive_pole_token_representations, dim=0)
            negative_pole_token_representations = torch.cat(negative_pole_token_representations, dim=0)

            # TODO: add comment
            positive_pole_token_representations_mean = positive_pole_token_representations.mean(dim=0).unsqueeze(0)
            positive_pole_token_representations_mean = positive_pole_token_representations_mean / \
                torch.norm(positive_pole_token_representations_mean, dim=1, keepdim=True)

            # TODO: add comment
            negative_pole_token_representations_mean = negative_pole_token_representations.mean(dim=0).unsqueeze(0)
            negative_pole_token_representations_mean = negative_pole_token_representations_mean / \
                torch.norm(negative_pole_token_representations_mean, dim=1, keepdim=True)

            # TODO: add comment
            dipole_features_token_mean.append(torch.cat([positive_pole_token_representations_mean,
                                                         negative_pole_token_representations_mean], dim=0).unsqueeze(0))

            # Normalise dipole token representations
            positive_pole_token_representations = positive_pole_token_representations / \
                torch.norm(positive_pole_token_representations, dim=1, keepdim=True)
            negative_pole_token_representations = negative_pole_token_representations / \
                torch.norm(negative_pole_token_representations, dim=1, keepdim=True)

            # Get the logarithmic map of the (normalised) token representations onto the tangent space at the CLS dipole
            # representations
            positive_pole_token_representations = self.logarithmic_map(s=positive_pole_token_representations_mean,
                                                                       q=positive_pole_token_representations)
            negative_pole_token_representations = self.logarithmic_map(s=negative_pole_token_representations_mean,
                                                                       q=negative_pole_token_representations)

            # Calculate sample covariances of the projected token representations
            positive_pole_token_representations_cov_diag = torch.diag(torch.cov(positive_pole_token_representations.T))
            negative_pole_token_representations_cov_diag = torch.diag(torch.cov(negative_pole_token_representations.T))

            dipole_features_token_cov.append(torch.cat([positive_pole_token_representations_cov_diag.unsqueeze(0),
                                                        negative_pole_token_representations_cov_diag.unsqueeze(0)],
                                                       dim=0).unsqueeze(0))

        return torch.cat(dipole_features_cls, dim=0), \
            torch.cat(dipole_features_token_mean, dim=0), \
            torch.cat(dipole_features_token_cov, dim=0)
