# Hyperparameter study

One factor varied at a time from the baseline, repeated over seeds.
A difference smaller than the standard deviation is not a result.

## optimizer

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| sgd_momentum | 0.3517 | 0.0000 | 3 | 51,018 |
| adamw | 0.7483 **<-** | 0.0000 | 3 | 51,018 |

## learning_rate

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| 0.001 | 0.7483 | 0.0000 | 3 | 51,018 |
| 0.01 | 0.8483 **<-** | 0.0000 | 3 | 51,018 |

## batch_size

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| 128 | 0.8183 **<-** | 0.0000 | 3 | 51,018 |
| 256 | 0.7483 | 0.0000 | 3 | 51,018 |

## activation

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| relu | 0.7483 | 0.0000 | 3 | 51,018 |
| tanh | 0.7650 **<-** | 0.0000 | 3 | 51,018 |

## dropout

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| 0.0 | 0.7517 **<-** | 0.0000 | 3 | 51,018 |
| 0.3 | 0.7400 | 0.0000 | 3 | 51,018 |

## hidden_sizes

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| 32 | 0.5800 | 0.0000 | 3 | 25,514 |
| 64-32 | 0.6733 **<-** | 0.0000 | 3 | 52,842 |

## loss

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| cross_entropy | 0.7483 **<-** | 0.0000 | 3 | 51,018 |
| mse | 0.7200 | 0.0000 | 3 | 51,018 |

## scheduler

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| none | 0.7483 **<-** | 0.0000 | 3 | 51,018 |
| cosine | 0.7050 | 0.0000 | 3 | 51,018 |

## weight_decay

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| 0.0 | 0.7483 **<-** | 0.0000 | 3 | 51,018 |
| 0.001 | 0.7483 **<-** | 0.0000 | 3 | 51,018 |

## augment

| level | val accuracy | std | epochs | params |
|---|---|---|---|---|
| False | 0.7483 **<-** | 0.0000 | 3 | 51,018 |
| True | 0.7000 | 0.0000 | 3 | 51,018 |
