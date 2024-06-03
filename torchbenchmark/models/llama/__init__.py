# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.



from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import ModelArgs, Transformer
import torch
from pathlib import Path


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 32
    
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        torch.set_default_device(device)
        self.device = device
        self.model_args = ModelArgs(vocab_size=32000,device=device)
        with torch.device('meta'):
            self.model = Transformer.from_name("meta-llama/Llama-2-7b-hf")
        checkpoint_path = Path(r"torchbenchmark\models\llama\.data\checkpoints\meta-llama\Llama-2-7b-hf\model.pth")
        print(checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        self.model.config.device = device
        self.model.load_state_dict(checkpoint, assign=True)
        self.model = self.model.to(device)
        #self.batch_size = 1
        self.seq_len = 32
        self.example_inputs = (torch.ones([self.batch_size, self.seq_len], dtype=torch.int).to(device), torch.arange(self.seq_len).to(torch.int))
        
    def get_module(self):
        return self.model, self.example_inputs
    
    def train(self):
        error_msg = """
            As of March 6, 2023
            The weights for this model are not publicly available and require a valid research reason to use
            The publicly available github repo is inference only
            https://github.com/facebookresearch/llama
        """
        return NotImplementedError(error_msg)

    def eval(self):
        self.model.eval()
        self.model.setup_caches(self.batch_size, 64)
        out=self.model(*self.example_inputs)
        return (out,)
