class StepLR:
    def __init__(self, decay_rate, decay_interval=1):
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.target = decay_interval

    def forward(self, learning_rate, epoch):
        if epoch > self.target:
            learning_rate *= (1 - self.decay_rate)
            self.target += self.decay_interval
            
        return learning_rate 

class ExponentialDecay:
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate 

    def forward(self, learning_rate, epoch):
        return learning_rate * (self.decay_rate ** epoch)

class InverseTimeDecay:
    def __init__(self, decay_rate, decay_interval=1):
        self.decay_rate = decay_rate 
        self.decay_interval = decay_interval

    def forward(self, learning_rate, epoch):
        return learning_rate / (1 + self.decay_rate * (epoch // self.decay_interval))

class CosineAnnealingDecay:
    def __init__(self, initial_lr, min_lr, max_epochs):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs

    def forward(self, learning_rate, epoch):
        return self.min_lr + (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2