# Multi-layer perceptron versus a convolutional network

Same data, same split, same training loop, same epoch budget.
The only difference is how the first layers read the image.

| model | parameters | val accuracy | test accuracy | seconds |
|---|---|---|---|---|
| MLP [512, 512, 512, 512] | 1,199,114 | 0.9794 | 0.9826 | 47.8 |
| CNN 32-64 | 421,834 | 0.9913 | 0.9907 | 93.8 |
