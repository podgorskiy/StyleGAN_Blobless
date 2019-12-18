# Style GAN without blobs and without retraining

This is work of my unofficial Pytorch implementation (https://github.com/podgorskiy/StyleGan) of Style GAN paper **"A Style-Based Generator Architecture for Generative Adversarial Networks"**
https://arxiv.org/pdf/1812.04948.pdf

After cloning don't forget to run

```shell
git submodule update --init
```

To install requirements:

```python
pip install -r requirements.txt
```

First, you need to download model (karras2019stylegan-ffhq-1024x1024.pkl), or take your pretrainied model and put it in the root

Then, you need to convert tensorflow model to pytorch one:
```python
python convertor.py
```

After that, you can run:

```python
python Sample.py
```

If you want to run a custom trained model, you might want to adjust /configs/experiment_stylegan.yaml as well as convertor.py

