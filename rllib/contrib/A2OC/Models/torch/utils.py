from ray.rllib.utils.framework import try_import_torch
_, nn = try_import_torch()

# View layer abstraction so I can use Sequential
class View(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.view(-1, *self.sizes)  # i.e., -1, num_options, num_outputs