import os
import os.path as osp
import argparse
import torch
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from joblib import dump


def main():
    """Fit GMM models on VL features found in `experiments/gan_clip_features/`.

    Options:
        -v, --verbose     : set verbose mode on
        --n-components    : set number of mixture components
        --covariance-type : type of covariance parameters to use. Must be one of:
                                'full'      : each component has its own general covariance matrix.
                                'diag'      : each component has its own diagonal covariance matrix.
                                'spherical' : each component has its own single variance.

        --init-params     : method used to initialize the weights, the means and the precisions. String must be one of
                                'kmeans'           : responsibilities are initialized using kmeans.
                                'k-means++'        : use the k-means++ method to initialize.
                                'random'           : responsibilities are initialized randomly.
                                'random_from_data' : initial means are randomly selected data points.

    See also: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    """
    parser = argparse.ArgumentParser(description="Fix GMM models")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--n-components', type=int, default=512, help="number of mixture components")
    parser.add_argument('--covariance-type', type=str, default='full', choices=('spherical', 'diag', 'full'),
                        help="type of covariance parameters to use")
    parser.add_argument('--init-params', type=str, default='k-means++',
                        choices=('kmeans', 'k-means++', 'random', 'random_from_data'),
                        help="method used to initialize the weights, the means and the precisions")
    # Parse given arguments
    args = parser.parse_args()

    # Get features files
    input_dir = osp.join('experiments', 'gan_vl_features')
    features_file_list = [dI for dI in os.listdir(input_dir)
                          if os.path.isfile(osp.join(input_dir, dI)) and osp.join(input_dir, dI).endswith('.pt')]

    # Fit and store a GMM model for each feature file
    for feat_file in tqdm(features_file_list, desc='Progress: '):
        # Load features
        print("#. Load features {}...".format(feat_file))
        clip_features_file = osp.join(input_dir, feat_file)
        clip_features = torch.load(clip_features_file).to('cpu')

        # Fit GMM
        gm = GaussianMixture(n_components=args.n_components,
                             covariance_type=args.covariance_type,
                             n_init=3,
                             init_params=args.init_params,
                             verbose=args.verbose).fit(clip_features)

        # Save GMM model
        print("#. Save GMM model...")
        gmm_file = osp.join(input_dir, '{}_C-{}_{}.joblib'.format(osp.splitext(feat_file)[0],
                                                                  args.n_components,
                                                                  args.covariance_type))
        dump(gm, gmm_file)


if __name__ == '__main__':
    main()
