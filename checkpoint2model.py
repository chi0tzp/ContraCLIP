import argparse
import os.path as osp
import torch


def main():
    """An auxiliary script for converting a checkpoint file (`checkpoint.pt`) into a support sets (`support_sets.pt`)
    and a reconstructor (`reconstructor.pt`) weights files.

    Options:
        ================================================================================================================
        --exp : set experiment's wip model dir, as created by `train.py`, i.e., it should contain a sub-directory
                `models/` with a checkpoint file (`checkpoint.pt`). Checkpoint file contains the weights of the
                 support sets and the reconstructor at an intermediate stage of training (after a given iteration).
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Convert a checkpoint file into a support sets and a reconstructor "
                                                 "weights files")
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")

    # Parse given arguments
    args = parser.parse_args()

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))
    checkpoint_file = osp.join(models_dir, 'checkpoint.pt')
    if not osp.isfile(checkpoint_file):
        raise FileNotFoundError("Checkpoint file not found: {}".format(checkpoint_file))

    print("#. Convert checkpoint file into support sets and reconstructor weight files...")

    # Load checkpoint file
    checkpoint_dict = torch.load(checkpoint_file)
    print("  \\__Checkpoint dictionary: {}".format(checkpoint_dict.keys()))

    # Get checkpoint iteration
    checkpoint_iter = checkpoint_dict['iter']
    print("  \\__Checkpoint iteration: {}".format(checkpoint_iter))

    # Save latent support sets (LSS) weights file
    print("  \\__Save checkpoint latent support sets LSS weights file...")
    torch.save(checkpoint_dict['latent_support_sets'],
               osp.join(models_dir, 'latent_support_sets-{:07d}.pt'.format(checkpoint_iter)))


if __name__ == '__main__':
    main()
