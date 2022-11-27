# Parameter-Efficient Masking Networks (PEMN)

This repository is for our NeurIPS'22 paper:
> Parameter-Efficient Masking Network, [PDF](https://arxiv.org/abs/2210.06699), [arXiv](https://arxiv.org/abs/2210.06699), [Project Homepage](https://yueb17.github.io/PEMN/)\
> [Yue Bai](https://yueb17.github.io/), [Huan Wang](http://huanwang.tech/), [Xu Ma](https://ma-xu.github.io/), [Yitian Zhang](https://bespontaneous.github.io/homepage/), [Zhiqiang Tao](http://ztao.cc/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/)

PEMN explores the representative potential of random initialized parameters with limited unique values by learning diverse masks to deliver different feature mappings. We propose to use three parameter-efficient strategies: One-layer, Max-layer padding (MP), and Random vector padding (RP) to construct a random network based on a set of given random parameters, which is named as prototype. This exploration promises us a network can be efficiently represented as a small set of random values with a bunch of masks. Inspired by our exploration, we naturally propsoe a new paradigm for network compression for efficient network storage and transfer.

## Run
To train Random Padding (RP) strategy with 1e-3 ratio on CIFAR10 using ConvMixer with 6-block and 256-dim:

```
CUDA_VISIBLE_DEVICES=0 python train.py --parallel --pipe train \
                                --lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0.2 --reprob=0.2 \
                                --act gelu --epochs 100 --hdim 256 \
                                --save --save_file test.csv \
                                --depth 6 --model_type 'mask' --clone_type 'augment' --MP_RP_ratio 0.001 \
                                --exp_id 1
```

## Reference
Please cite this in your publication if our work helps your research. Should you have any questions, welcome to reach out to Yue Bai (bai.yue@northeastern.edu).

```
@article{bai2022parameter,
  title={Parameter-Efficient Masking Networks},
  author={Bai, Yue and Wang, Huan and Ma, Xu and Zhang, Yitian and Tao, Zhiqiang and Fu, Yun},
  journal={arXiv preprint arXiv:2210.06699},
  year={2022}
}
```


### Keep updating


