from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """LR = Initial_LR * (1 - iter / max_iter)^0.9"""
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        lr = [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]

        # Numerical stability for FP16 training
        if type(lr[0]) == complex:
            lr = [1e-7]
        return lr if lr[0] >= 1e-7 else [1e-7]