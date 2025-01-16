# Following [this](https://youtu.be/HS3Q_90hnDg?si=d1G4AWmbN778bLqS) video to build a u-net

## Training (Single vs Parallel)
- main.py is single GPU
- main_parallel is 2x GPU (2x 3090 in my case)

### Single


### Parallel (vs Single)
- Gradient synchronization is used, where each batch is processed on each GPU independently, then the gradients from each GPU are averaged, and the new weight set is set on each GPU for the next batch. Each GPU sends it's gradients to the other so the averaging happens in parallel, then the shared weights are updated in parallel as well. The sharing is done through Nvidia Collective Communications Library (NCCL).

