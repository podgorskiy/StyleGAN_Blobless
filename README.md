# Style GAN

Unofficial Pytorch implementation of Style GAN paper **"A Style-Based Generator Architecture for Generative Adversarial Networks"**
https://arxiv.org/pdf/1812.04948.pdf

https://arxiv.org/pdf/1710.10196.pdf

Original Tensorflow code:

https://github.com/NVlabs/stylegan


Generation example (4 x Titan X for 8 hours):
<div>
	<img src='/generation.jpg'>
</div>

To install requirements:

```python
pip install -r requirements.txt
```

To download and prepare dataset:
```python
python prepare_celeba.py
python downscale_celeba.py
```

To train:
```python
python StyleGAN.py
```
