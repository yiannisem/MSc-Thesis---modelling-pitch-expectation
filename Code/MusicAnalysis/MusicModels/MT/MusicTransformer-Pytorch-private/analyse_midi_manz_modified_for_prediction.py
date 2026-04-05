# run from this directory: MSc Thesis/Kern Music Transformer/Code/MusicAnalysis/MusicModels/MT

import torch
import numpy as np
import pandas as pd
import os
from glob import glob

from third_party.midi_processor.processor import encode_midi
from utilities.argument_funcs import parse_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano_analysis import process_midi
from utilities.constants import TORCH_LABEL_TYPE
from utilities.device import get_device

# helper 
def norm_neg_log_probs(neg_log_probs):
    probs = np.exp(-neg_log_probs)
    probs = probs / probs.sum(axis=1)[:, None]
    return -np.log(probs)

# setup 
study = 'TRIAL'  # not really used now
dir_cur = os.path.dirname(__file__)
dir_base = dir_cur[:dir_cur.find('Code')]

dir_midi = os.path.join(dir_cur, 'manz_midi_stimuli')
dir_out = os.path.join(dir_cur, 'probe_prediction_results')
os.makedirs(dir_out, exist_ok=True)

midi_paths = sorted(glob(os.path.join(dir_midi, '*.mid')) +
                    glob(os.path.join(dir_midi, '*.midi')))
filenames = [os.path.basename(p) for p in midi_paths]

model_tag = 'transformer'
len_context_list = [1] # overwritten by args.primer_len = 10**9

for len_context in len_context_list:
    fn_out = f'rpr_ftuned_manz_{model_tag}_all_probs.csv'
    fn_out_raw = f'rpr_ftuned_manz_{model_tag}_all_probs_raw.csv'

    probs_rows = []
    raw_probs_rows = []
    first_row_idx_for_file = []
    actual_note_rows = []  # (filename, note_index, actual_pitch, probability)

    args = parse_generate_args()
    args.model_weights = os.path.abspath(os.path.join(
        dir_cur, '..', 'TrainedModels', 'MTMaestroRPR_finetuned_on_MCCC',
        'results', 'best_loss_weights.pickle'
    ))
    args.rpr = True
    args.max_sequence = 2048
    # ensure full history context at each step
    args.primer_len = 10**9  # so i_note < args.primer_len holds

    # Build & load model once
    mt_model = MusicTransformer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence,
        rpr=args.rpr
    ).to(get_device())
    mt_model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))
    mt_model.eval()

    i_row = 0

    # per file per note loop
    for fn in filenames:
        args.primer_file = os.path.join(dir_midi, fn)
        raw_mid = encode_midi(args.primer_file) # list containing flat token sequence (converted from MIDI)

        # indices of pitch tokens only (0..127)
        note_positions = [i for i, j in enumerate(raw_mid) if j <= 127]
        if len(note_positions) == 0:
            # skip empty files defensively
            continue

        first_row_idx_for_file.append(i_row)

        for i_note, note_pos in enumerate(note_positions):
            if i_note < args.primer_len: # always holds while args.primer_len is chosen to be larger than the number of notes in the piece
                sequence_start = 0
                args.num_prime = note_pos
            else:
                sequence_start = note_positions[i_note - args.primer_len]
                args.num_prime = note_pos - sequence_start

            # the primer is the sequence of tokens in the context before the note we want to predict
            # it returns the input sequence of length args.num_prime and the target sequence (only used for training)
            primer, _ = process_midi(raw_mid, args.num_prime, sequence_start, random_seq=False)
            primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device()) # convert primer to pytorch tensor
            args.target_seq_length = args.num_prime + 1 # length of the sequence to predict (primer + one note)

            with torch.set_grad_enabled(False):
                # uses get_probs method of MusicTransformer class to get the probability distribution of the next note
                probs, raw_probs = mt_model.get_probs(
                    primer, i_note, target_seq_length=args.target_seq_length, raw_mid=raw_mid
                )

            probs_rows.append(probs)
            raw_probs_rows.append(raw_probs)

            # Record the probability assigned to the actual pitch heard
            actual_pitch = raw_mid[note_pos]
            actual_prob = raw_probs[actual_pitch]
            actual_note_rows.append((fn, i_note, actual_pitch, actual_prob))


            print(f"Event {i_note+1} / {len(note_positions)} in composition {fn} "
                  f"({filenames.index(fn)+1}/{len(filenames)})")
            i_row += 1

    # Stack and save existing outputs
    if len(probs_rows) > 0:
        probs_data = np.vstack(probs_rows)
        raw_probs_data = np.vstack(raw_probs_rows)

        probs_data = norm_neg_log_probs(probs_data)
        for idx in first_row_idx_for_file:
            probs_data[idx, :] = -np.log(np.repeat(1/128, 128))

        headers = list(range(128))
        pd.DataFrame(probs_data, columns=headers).to_csv(
            os.path.join(dir_out, fn_out), index=False
        )
        pd.DataFrame(raw_probs_data, columns=headers).to_csv(
            os.path.join(dir_out, fn_out_raw), index=False
        )

        # Also save pitch-only probabilities normalised to sum=1
        row_sums = raw_probs_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero
        raw_probs_norm = raw_probs_data / row_sums

        pd.DataFrame(raw_probs_norm, columns=headers).to_csv(
            os.path.join(dir_out, f"rpr_ftuned_manz_{model_tag}_all_probs_norm.csv"),
            index=False
        )

        # Save probabilities for MIDI range 60-79 only (re-normalised)
        midi_range = list(range(60, 80)) # column labels
        raw_probs_range = raw_probs_data[:, 60:80]
        range_sums = raw_probs_range.sum(axis=1, keepdims=True)
        range_sums[range_sums == 0] = 1.0 # possibly not needed
        raw_probs_range_norm = raw_probs_range / range_sums

        pd.DataFrame(raw_probs_range_norm, columns=midi_range).to_csv(
            os.path.join(dir_out, f"rpr_ftuned_manz_{model_tag}_probs_60_to_79.csv"),
            index=False
        )

    # Save per-note actual pitch expectations
    if len(actual_note_rows) > 0:
        df_actual = pd.DataFrame(
            actual_note_rows,
            columns=["filename", "note_index", "actual_pitch", "probability"]
        )
        df_actual.to_csv(
            os.path.join(dir_out, f"rpr_ftuned_manz_{model_tag}_actual_note_probs.csv"),
            index=False
        )
