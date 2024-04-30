# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/micorosoft/phi-2"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.dense.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.dense.bias": "layers.{}.attention.wo.bias",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.fc1.weight': 'layers.{}.feed_forward.w1.weight',
        'model.layers.{}.mlp.fc1.bias': 'layers.{}.feed_forward.w1.bias',
        "model.layers.{}.mlp.fc2.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.mlp.fc2.bias": "layers.{}.feed_forward.w2.bias",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.input_layernorm.bias": "layers.{}.attention_norm.bias",
        "model.final_layernorm.weight": "norm.weight",
        "model.final_layernorm.bias": "norm.bias",
        "lm_head.weight": "output.weight",
        "lm_head.bias": "output.bias",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        print(w.shape, (n_head, 2, config.head_dim // 2, dim) )
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = load_file(str(file))
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        print(key)
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key, count=1)
            layer_num = re.search(r'\d+', key).group(0)

            new_key = weight_map[abstract_key]

            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq.weight" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

        if "wq.bias" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/microsoft/phi-2"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
