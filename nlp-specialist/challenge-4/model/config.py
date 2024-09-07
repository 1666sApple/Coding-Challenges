import torch

class Config:
    def __init__(self):
        self.model_name = 'bert-base-multilingual-cased'
        self.max_length = 128
        self.batch_size = 4
        self.num_epochs = 15
        self.learning_rate = 1e-3
        self.weight_decay = 0.1
        self.gradient_clip_value = 1.0
        self.accumulation_steps = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
