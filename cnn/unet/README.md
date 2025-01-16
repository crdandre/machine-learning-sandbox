# Following [this](https://youtu.be/HS3Q_90hnDg?si=d1G4AWmbN778bLqS) video to build a u-net and expanding on it
- adding compatibility with remote servers
- parallel training (data parallelism)
- [also great](https://www.youtube.com/watch?v=NhdzGfB1q74)

## Training (Single vs Parallel)
- main.py is single GPU
- main_parallel is 2x GPU (2x 3090 in my case)

### Single
- ye olde load weights and biases, forward pass, measure loss, backprop the gradient

### Parallel (vs Single)
- Gradient synchronization is used, where each batch is processed on each GPU independently, then the gradients from each GPU are averaged, and the new weight set is set on each GPU for the next batch. Each GPU sends it's gradients to the other so the averaging happens in parallel, then the shared weights are updated in parallel as well. The sharing is done through Nvidia Collective Communications Library (NCCL).

- Consider matching the effective batch size for both? eh. This would favor single probably since there is the same data with more overhead.