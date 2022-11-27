# Parameter-Efficient Masking Networks (PEMN)

This repository is for our NeurIPS'22 paper:
> Parameter-Efficient Masking Network, [PDF](https://arxiv.org/abs/2210.06699), [arXiv](https://arxiv.org/abs/2210.06699), [Project Homepage](https://yueb17.github.io/PEMN/)\
> [Yue Bai](https://yueb17.github.io/), [Huan Wang](http://huanwang.tech/), [Xu Ma](https://ma-xu.github.io/), [Yitian Zhang](https://bespontaneous.github.io/homepage/), [Zhiqiang Tao](http://ztao.cc/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/)

PEMN explores the representative potential of random initialized parameters with limited unique values by learning diverse masks to deliver different feature mappings. We propose to use three parameter-efficient strategies: One-layer, Max-layer padding (MP), and Random vector padding (RP) to construct a random network based on a set of given random parameters, which is named as prototype. This exploration promises us a network can be efficiently represented as a small set of random values with a bunch of masks. Inspired by our exploration, we naturally propsoe a new paradigm for network compression for efficient network storage and transfer.

To train ConvMixer with 6-block and 256-dim:
	code

## Updating
