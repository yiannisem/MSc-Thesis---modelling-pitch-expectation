import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# make sure we import numpy
import numpy as np

# another comment
# and another one

## setup args and model, load primer

args = parse_generate_args()

args.model_weights = '/project/3018045.02/trained_models/best_acc_weights.pickle'
args.primer_file = '/project/3018045.01/pius-scratch/custommidi/raindrop-begin.mid'
args.num_prime = 240
args.target_seq_length = 1200
args.rpr = True
args.output_dir = 'output'

print_generate_args(args)

# use prime from dataset
# _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)
# idx = round(np.random.rand()*len(dataset))
# primer, _  = dataset[idx]
# primer = primer.to(get_device())

raw_mid = encode_midi(args.primer_file)
primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
            d_model=args.d_model, dim_feedforward=args.dim_feedforward,
            max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

model.load_state_dict(torch.load(args.model_weights))

## optionally save primer

# Saving primer first
f_path = os.path.join(args.output_dir, "primer.mid")
decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)


## random sampling generation

model.eval()
with torch.set_grad_enabled(False):
    rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0,
        top_p=0.999)

## save to disk

f_path = os.path.join(args.output_dir, "rand.mid")
decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)