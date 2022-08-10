import torch
import clip


class SemanticDipoles:
    def __init__(self, corpus, clip_model, use_cuda):
        self.corpus = corpus
        self.use_cuda = use_cuda
        self.clip_model = clip_model.to('cuda' if self.use_cuda else 'cpu')
        self.num_dipoles = len(self.corpus)
        self.dim = 512

    def get_prompt_features(self):
        # Get CLIP text features for the given dipoles
        prompt_features = []
        for t in range(self.num_dipoles):
            # Get dipole's CLIP text representations
            dipole_features = self.clip_model.encode_text(
                clip.tokenize(self.corpus[t]).to('cuda' if self.use_cuda else 'cpu'))
            # Normalize dipole's features
            dipole_features /= torch.norm(dipole_features, dim=1, keepdim=True)
            prompt_features.append(dipole_features.unsqueeze(0))

        return torch.cat(prompt_features, dim=0)