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

dir_midi = os.path.join(dir_cur, 'cudlun_midi_stimuli_with_probe')
dir_out = os.path.join(dir_cur, 'probe_prediction_results')
os.makedirs(dir_out, exist_ok=True)

midi_paths = sorted(glob(os.path.join(dir_midi, '*.mid')) +
                    glob(os.path.join(dir_midi, '*.midi')))
filenames = [os.path.basename(p) for p in midi_paths]

model_tag = 'transformer'
len_context_list = [1]

for len_context in len_context_list:
    fn_out = f'rpr_ftuned_cudlun_{model_tag}_all_probs.csv'
    fn_out_raw = f'rpr_ftuned_cudlun_{model_tag}_all_probs_raw.csv'

    probs_rows = []
    raw_probs_rows = []
    first_row_idx_for_file = []

    # NEW: containers for probe-only output
    probe_rows = []
    probe_filenames = []

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

    for fn in filenames:
        args.primer_file = os.path.join(dir_midi, fn)
        raw_mid = encode_midi(args.primer_file)

        # indices of pitch tokens (0..127)
        note_positions = [i for i, j in enumerate(raw_mid) if j <= 127]
        if len(note_positions) == 0:
            # skip empty files defensively
            continue

        first_row_idx_for_file.append(i_row)

        for i_note, note_pos in enumerate(note_positions):
            if i_note < args.primer_len:
                sequence_start = 0
                args.num_prime = note_pos
            else:
                sequence_start = note_positions[i_note - args.primer_len]
                args.num_prime = note_pos - sequence_start

            primer, _ = process_midi(raw_mid, args.num_prime, sequence_start, random_seq=False)
            primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())
            args.target_seq_length = args.num_prime + 1

            with torch.set_grad_enabled(False):
                probs, raw_probs = mt_model.get_probs(
                    primer, i_note, target_seq_length=args.target_seq_length, raw_mid=raw_mid
                )

            probs_rows.append(probs)
            raw_probs_rows.append(raw_probs)

            # Capture the final note's distribution (the probe note you appended)
            if i_note == len(note_positions) - 1:
                rp = np.asarray(raw_probs, dtype=np.float64)
                s = rp.sum()
                if s == 0:
                    s = 1.0
                probe_rows.append(rp / s)   # normalize to sum=1
                # probe_filenames.append(fn)
                clean_name = fn.replace("_with_probe", "")
                probe_filenames.append(clean_name)


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
            os.path.join(dir_out, f"rpr_ftuned_cudlun_{model_tag}_all_probs_norm.csv"),
            index=False
        )

    # NEW: save probe-only CSV (one row per input file; filename + prob_0..prob_127)
    if len(probe_rows) > 0:
        probe_mat = np.vstack(probe_rows)
        probe_cols = [f"prob_{i}" for i in range(128)]
        df_probe = pd.DataFrame(probe_mat, columns=probe_cols)
        df_probe.insert(0, "filename", probe_filenames)
        out_probe = os.path.join(dir_out, f"rpr_ftuned_cudlun_{model_tag}_probe_probs.csv")
        df_probe.to_csv(out_probe, index=False)
        print(f"Saved probe-only distributions to: {out_probe}")
    else:
        print("No probe rows captured; check input MIDIs.")
