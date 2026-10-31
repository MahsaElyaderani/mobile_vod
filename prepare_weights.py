"""Prepare original TF weights or Google's Hugging Face conversion for this model."""
from path import parse_paths

import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import torch
from models.mobilenet_v1 import MobileNetV1Backbone
from utils.pretrained import (HF_MODEL, HF_REVISION, OFFICIAL_URL, sha256_file,
                              map_huggingface_state, map_tensorflow_checkpoint)
from utils.runtime import atomic_save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['official-tf', 'huggingface'], default='huggingface')
    parser.add_argument('--checkpoint', help='Local HF .bin file or TensorFlow checkpoint prefix')
    parser.add_argument('--output', default=None)
    args = parse_paths(parser, "prepare_weights")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    backbone = MobileNetV1Backbone()
    with tempfile.TemporaryDirectory() as folder:
        if args.source == 'huggingface':
            url = f'https://huggingface.co/{HF_MODEL}/resolve/{HF_REVISION}/pytorch_model.bin'
            path = Path(args.checkpoint) if args.checkpoint else Path(folder) / 'pytorch_model.bin'
            if not args.checkpoint:
                urllib.request.urlretrieve(url, path)
            # Hash verification also prevents local files being mislabeled as Google's release.
            digest = sha256_file(path)
            from utils.pretrained import HF_SHA256
            if digest != HF_SHA256:
                raise ValueError('HF checkpoint SHA256 does not match the verified Google release.')
            state = map_huggingface_state(backbone, torch.load(path, weights_only=True, map_location='cpu'))
            source = {'type': 'huggingface_conversion', 'model': HF_MODEL, 'revision': HF_REVISION,
                      'sha256': digest, 'url': url}
        else:
            if args.checkpoint:
                prefix = Path(args.checkpoint)
                source = {'type': 'tensorflow_local', 'checkpoint_prefix': str(prefix),
                          'index_sha256': sha256_file(str(prefix) + '.index')}
            else:
                archive = Path(folder) / 'official.tgz'
                urllib.request.urlretrieve(OFFICIAL_URL, archive)
                digest = sha256_file(archive)
                from utils.pretrained import TF_ARCHIVE_SHA256
                if digest != TF_ARCHIVE_SHA256:
                    raise ValueError('Original archive SHA256 does not match the verified release.')
                # Copy only checkpoint files, never extract arbitrary archive paths.
                with tarfile.open(archive, 'r:gz') as tar:
                    for member in tar:
                        name = Path(member.name).name
                        if member.isfile() and name.startswith('mobilenet_v1_1.0_224.ckpt.'):
                            with tar.extractfile(member) as src, open(Path(folder) / name, 'wb') as dst:
                                dst.write(src.read())
                prefix = Path(folder) / 'mobilenet_v1_1.0_224.ckpt'
                source = {'type': 'tensorflow_original', 'url': OFFICIAL_URL, 'sha256': digest}
            state = map_tensorflow_checkpoint(backbone, prefix)
        atomic_save_checkpoint({'kind': 'mobilenet_v1_backbone', 'padding': 'tf_same',
                                'backbone': state, 'source': source}, output, overwrite=False)
    print(f'Saved all {len(state)} backbone tensors to {output}')


if __name__ == '__main__':
    main()
