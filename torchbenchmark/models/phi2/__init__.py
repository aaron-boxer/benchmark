# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch_directml
import torch

device = torch_directml.device(torch_directml.default_device())

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from .model import Transformer

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)

        import argparse
        parser = argparse.ArgumentParser(description='Your CLI description.')

        parser.add_argument('--prompt', type=str, default="Hello, my name is ", help='Input prompt.')
        parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
        parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
        parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens.')
        parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
        parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
        parser.add_argument('--checkpoint_path', type=Path, default=Path("torchbenchmark/models/phi2/.data/checkpoints/microsoft/phi2/model.pth"), help='Model checkpoint path.')
        parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
        parser.add_argument('--precision', type=str, default='float32', help='Model Inference precision.')

        #args = parser.parse_args()
        #self.main(
        #    args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        #    args.temperature, args.checkpoint_path, args.profile, args.precision
        #)
        self.main("Hello my name is Frank")

        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        self.model.train()
        targets = self.example_target
        output = self.model(self.images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(self.images)
        return (out,)

    def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
        probs = logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
        # input_pos: [B, S]
        logits = model(x, input_pos)
        return sample(logits, **sampling_kwargs)[0]

    def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = model(x, input_pos)
        return sample(logits, **sampling_kwargs)

    def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
                next_token, next_prob = decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                callback(new_tokens[-1])
                new_probs.append(next_prob.clone())
                cur_token = next_token.view(1, -1)

        return new_tokens, new_probs


    def model_forward(model, x, input_pos):
        return model(x, input_pos)

    @torch.no_grad()
    def generate(
        model: Transformer,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        interactive: bool,
        callback = lambda x: x,
        precision=None,
        **sampling_kwargs
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """

        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens
        if interactive:
            max_seq_length = 350
        else:
            max_seq_length = min(T_new, model.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length, dtype=precision)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[T + 1:] = torch.cat(generated_tokens)

        return seq

    def encode_tokens(tokenizer, string, bos=False, device='cuda'):
        tokens = tokenizer.encode(string)
        if bos:
            tokens = [tokenizer.encode(tokenizer.bos_token)[0]] + tokens
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def _load_model(self, checkpoint_path, device, precision, use_tp):
        with torch.device('meta'):
            self.model = Transformer.from_name(checkpoint_path.parent.name)

        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        self.model.load_state_dict(checkpoint, assign=True)
        
        self.model = self.model.to(device=device, dtype=precision)
        return self.model.eval()

    B_INST, E_INST = "[INST]", "[/INST]"

    def main(self,
        prompt: str = "Hello, my name is",
        interactive: bool = False,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.8,
        checkpoint_path: Path = Path("torchbenchmark\models\phi2\.data\checkpoints\microsoft\phi-2\model.pth"),
        profile: Optional[Path] = None,
        precision: str = 'float32'
    ) -> None:
        """Generates text samples based on a pre-trained Transformer model and tokenizer.
        """
        global device
        assert checkpoint_path.is_file(), checkpoint_path

        # tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        # assert tokenizer_path.is_file(), tokenizer_path

        global print
        rank = None
        
        use_tp = rank is not None
        if use_tp:
            if rank != 0:
                print = lambda *args, **kwargs: None

        print(f"Using device={device}")
        #precision = torch.float32 if args.precision == 'float32' else torch.float16
        precision = torch.float32
        is_chat = "chat" in str(checkpoint_path)

        print("Loading model ...")
        t0 = time.time()
        self._load_model(checkpoint_path, device, precision, use_tp)

        print(f"Time to load model: {time.time() - t0:.02f} seconds")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        encoded = encode_tokens(tokenizer, prompt, bos=False, device=device)
        prompt_length = encoded.size(0)

        torch.manual_seed(1234)

        model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
        aggregate_metrics = {
            'tokens_per_sec': [],
            'accept_counts': [],
        }
        start = -1 if compile else 0

        for i in range(start, num_samples):
            if i >= 0 and interactive:
                prompt = input("What is your prompt? ")
                if is_chat:
                    prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                encoded = encode_tokens(tokenizer, prompt, bos=False, device=device)
                # encoded = tokenizer(prompt, return_attention_mask=False, return_tensors="pt")["input_ids"][0].to(torch.int32).cuda()

            callback = lambda x : x
            t0 = time.perf_counter()
            import contextlib
            if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile(record_shapes=True, with_stack=True, profile_memory=True)
            with prof:
                y = generate(
                    self.model,
                    encoded,
                    max_new_tokens,
                    interactive=interactive,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                    precision=precision
                )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            if hasattr(prof, "export_chrome_trace"):
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
                prof.export_chrome_trace(f"{profile}.json")
            t = time.perf_counter() - t0
            if not interactive:
                print(tokenizer.decode(y.tolist()))
            else:
                print()
            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            aggregate_metrics['tokens_per_sec'].append(tokens_sec)
            print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
            print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        print("==========")

        print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

