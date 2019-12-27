# Original Style GAN without blobs and without retraining
## Simple trick to remove blobs

![Demo](https://github.com/podgorskiy/StyleGAN_Blobless/raw/master/demo.png)

This work is based on my unofficial Pytorch implementation (https://github.com/podgorskiy/StyleGan) of Style GAN paper **"A Style-Based Generator Architecture for Generative Adversarial Networks"**
https://arxiv.org/pdf/1812.04948.pdf

After cloning don't forget to run

```shell
git submodule update --init
```

To install requirements:

```python
pip install -r requirements.txt
```

First, you will need to download model (karras2019stylegan-ffhq-1024x1024.pkl), or take your pretrainied model and put it in the root of the clonned repository.

Then, you will need to convert tensorflow model to pytorch one:
```python
python convertor.py
```

After that, you can run:

```python
python Sample.py
```

If you want to run a custom trained model, you might want to adjust /configs/experiment_stylegan.yaml as well as convertor.py

## How it works?

See https://github.com/podgorskiy/StyleGAN_Blobless/blob/master/net.py#L260
At resolution 64x64 (that is the resolution where the artifact is introduced), all entries of the tensor that are greater than 200 are pruned (assigned 0).
However, if we do so, we are going to effect statistics of the tensor and therefore will break all consiquent operations. To avoid that, two branches are executed simultaniusly: pruned one and the original one. The original one is used to compute AdaIn, coefficients which are then used to scale the pruned one.
