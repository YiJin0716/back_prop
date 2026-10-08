"""Identical V4 training, with the missing-nodule CT sampler enabled."""
from back_prop.model_v4.train import main

if __name__ == '__main__':
    main(oversample=True)
