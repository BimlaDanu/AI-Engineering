# Ensembling and test-time augmentation

Members differ only in their seed. The ensemble averages class
probabilities; TTA averages one model over augmented copies of the input.

| model | val accuracy | test accuracy |
|---|---|---|
| member seed=0 | 0.9802 | 0.9806 |
| member seed=1 | 0.9815 | 0.9815 |
| member seed=2 | 0.9807 | 0.9823 |
| ensemble of 3 | 0.9819 | 0.9835 |
| TTA x4 (first member) | 0.9711 | 0.9724 |
