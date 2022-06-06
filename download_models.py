import sys
import os
import os.path as osp
import hashlib
import tarfile
import time
import urllib.request
from lib import GENFORCE, GENFORCE_MODELS, ARCFACE, ContraCLIP_models


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r      \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def download(src, sha256sum, dest):
    tmp_tar = osp.join(dest, ".tmp.tar")
    try:
        urllib.request.urlretrieve(src, tmp_tar, reporthook)
    except:
        raise ConnectionError("Error: {}".format(src))

    sha256_hash = hashlib.sha256()
    with open(tmp_tar, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

        sha256_check = sha256_hash.hexdigest() == sha256sum
        print()
        print("      \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def main():
    """Download pre-trained GAN generators, ArcFace, and ContraCLIP models:
        -- GenForce GAN generators [1]
        -- ArcFace [2]
        -- ContraCLIP [3]

    References:
        [1] https://genforce.github.io/
        [2] Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition."
            Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
        [3] TODO: add ref
    """
    # Create pre-trained models root directory
    pretrained_models_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_models_root, exist_ok=True)

    print("#. Download pre-trained ArcFace model...")
    print("  \\__.ArcFace")
    if osp.exists(osp.join(pretrained_models_root, 'arcface', 'model_ir_se50.pth')):
        print("      \\__Already exists.")
    else:
        download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_models_root)

    # Download the following pre-trained GAN generators (under models/pretrained/)
    print("  \\__.GenForce")
    download_genforce_models = False
    for k, v in GENFORCE_MODELS.items():
        if not osp.exists(osp.join(pretrained_models_root, 'genforce', v[0])):
            download_genforce_models = True
            break
    if download_genforce_models:
        download(src=GENFORCE[0], sha256sum=GENFORCE[1], dest=pretrained_models_root)
    else:
        print("      \\__Already exists.")

    # Download pre-trained ContraCLIP models
    pretrained_contraclip_root = osp.join('experiments', 'complete')
    os.makedirs(pretrained_contraclip_root, exist_ok=True)

    print("#. Download pre-trained ContraCLIP models...")
    print("  \\__.ContraCLIP pre-trained models...")
    download(src=ContraCLIP_models[0],
             sha256sum=ContraCLIP_models[1],
             dest=pretrained_contraclip_root)


if __name__ == '__main__':
    main()
