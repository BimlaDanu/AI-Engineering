---
title: Generative Models
subject: dl
topic: generative models
difficulty: advanced
year: 2026
source: In-house study note
---

# Generative Models

A **discriminative** model learns $p(y \mid x)$ — a boundary between classes. A **generative**
model learns the data distribution $p(x)$ itself, so it can *sample new data* that looks like
the training set: images, audio, molecules, or text. Deep generative models power image
synthesis, super-resolution, data augmentation, and (as decoders) modern LLMs.

## Autoencoders and VAEs

An **autoencoder** compresses input through a bottleneck (**encoder** → latent code →
**decoder**) and is trained to reconstruct its input. The latent code is a learned, compact
representation — useful for denoising and anomaly detection — but a plain autoencoder's
latent space has "holes" you cannot sample from meaningfully.

A **Variational Autoencoder (VAE)** fixes this by making the encoder output a *distribution*
(mean and variance) and regularising it toward a standard normal via a KL term. Its loss
balances reconstruction against that regulariser (the **ELBO**):

$$\mathcal{L} = \underbrace{\mathbb{E}[\log p(x \mid z)]}_{\text{reconstruction}} -
\underbrace{D_{\mathrm{KL}}\!\big(q(z \mid x)\,\Vert\,p(z)\big)}_{\text{regulariser}}$$

The **reparameterisation trick** ($z = \mu + \sigma \odot \varepsilon$) keeps sampling
differentiable so gradients flow. VAEs give a smooth, sampleable latent space at the cost of
somewhat blurry samples.

## GANs

A **Generative Adversarial Network** pits two networks against each other: a **generator**
turns noise into fake samples, and a **discriminator** learns to tell real from fake. They
play a minimax game — the generator improves until its samples fool the discriminator. GANs
produce sharp, realistic images but are notoriously unstable to train and prone to **mode
collapse** (generating little variety). WGAN, spectral normalisation, and progressive growing
were introduced to stabilise them.

## Diffusion models

**Diffusion models** (the engine behind Stable Diffusion, DALL·E, Imagen) define a *forward
process* that gradually adds Gaussian noise to data over many steps until it is pure noise,
then train a network to *reverse* it — predicting and removing a little noise at each step.
Generation starts from random noise and denoises iteratively into a sample. They are stable
to train and produce state-of-the-art fidelity and diversity, at the cost of slow, multi-step
sampling (mitigated by DDIM, distillation, and latent diffusion).

## Autoregressive models

**Autoregressive** models factor $p(x) = \prod_t p(x_t \mid x_{<t})$ and generate one token
at a time conditioned on the past. This is exactly how GPT-style LLMs generate text — the
same principle as PixelCNN for images and WaveNet for audio. Sampling is inherently
sequential, which is why fast decoding is an active research area.
