---
title: Convolutional Neural Networks
subject: dl
topic: cnn
difficulty: intermediate
year: 2026
source: in-house study notes
---

# Convolutional neural networks (CNNs)

## Why not a plain dense network for images?

A 224×224 RGB image has ~150k input values. A dense layer connecting all of them to 1000
hidden units needs 150 million weights — and it would have to relearn "an edge" separately at
every pixel position. CNNs fix both problems with two ideas:

1. **Local connectivity** — each unit looks only at a small patch (e.g. 3×3).
2. **Weight sharing** — the same small filter slides across the whole image, so a feature
   detector learned once works everywhere (*translation equivariance*).

## The convolution operation

A filter $K$ slides over the input $X$ producing a feature map:

$$ (X * K)_{ij} = \sum_{m}\sum_{n} X_{i+m,\, j+n} \, K_{m,n} $$

Each convolutional layer learns many filters (e.g. 64), each producing its own feature map.
Key hyperparameters:

- **Kernel size** (3×3 is the modern default),
- **Stride** — step size of the slide; stride 2 halves spatial resolution,
- **Padding** — zeros added at borders so the output size matches the input.

## Pooling and hierarchy

**Max pooling** (take the maximum over each small window) downsamples feature maps, adding
a degree of translation *invariance* and cutting computation. A classic CNN alternates
convolution + ReLU blocks with pooling, so deeper layers see a larger **receptive field**:
early layers detect edges and textures, middle layers detect parts (eyes, wheels), late
layers detect whole objects. This learned hierarchy is exactly the feature engineering that
classical computer vision did by hand.

## Landmark architectures

- **LeNet-5 (1998)** — digits; proved the concept.
- **AlexNet (2012)** — won ImageNet by a large margin; ReLU + dropout + GPUs started the
  deep-learning era.
- **VGG (2014)** — showed stacked small 3×3 kernels beat larger ones.
- **ResNet (2015)** — residual connections $h = x + f(x)$ made 100+ layer networks trainable
  by letting gradients bypass layers.
- **Vision Transformers (ViT, 2020)** — replace convolutions with self-attention over image
  patches; now competitive when pretraining data is large, though CNNs retain a strong
  inductive bias for small-data regimes.

## Practical notes

- **Data augmentation** (crops, flips, colour jitter) is the cheapest regulariser for vision.
- **Transfer learning** — start from ImageNet-pretrained weights and fine-tune; with small
  datasets, freeze early layers and retrain only the head.
- **Batch normalisation** after convolutions stabilises and accelerates training.

## Key takeaways

- Convolution = local patches + shared weights → far fewer parameters and built-in
  translation equivariance.
- Depth builds a feature hierarchy: edges → parts → objects.
- Residual connections unlocked very deep vision models; ViTs are the attention-based
  alternative when data is plentiful.
