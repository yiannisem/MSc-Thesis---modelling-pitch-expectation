#%% Import packages, modules, functions
import torch
import os
import numpy as np
import pandas as pd

from third_party.midi_processor.processor import encode_midi

from utilities.argument_funcs import parse_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano_analysis import process_midi

from utilities.constants import *
from utilities.device import get_device

#%% Load data

def norm_neg_log_probs(neg_log_probs):
    
    probs = np.exp(-neg_log_probs)
    probs = probs/probs.sum(axis=1)[:,None]
    
    return -np.log(probs)

def MT_analyse(modality='EEG', len_context_list=[5]):

    study       = modality # 'MEG' or 'EEG'
    study       = 'MEG' # 'MEG' or 'EEG'
    dir_cur     = os.path.dirname(__file__)
    dir_base    = dir_cur[:dir_cur.find('Code')]
    dir_music   = dir_base + 'Data/MusicData/Music%s/' % study
    df          = pd.read_csv(dir_music + 'MidiData%s.csv' % study)
    dir_midi    = dir_base + 'Data/MusicData/Music%s/midi/' % study
    dir_out     = dir_base + 'Results/ResultsMusic/ResultsMusic%s/modelling/' % study
    filenames   = set(df['filename'])
    filenames   = sorted(list(filenames))
             
    model = 'MT'

    for len_context in len_context_list:
        
        fn_out = 'probs_%s_%d.csv' % (model, len_context)
        fn_out_raw = 'probs_%s_%d_raw.csv' % (model, len_context)
        
        # Initialize variables
        probs_data = np.empty(shape=(len(df),128))
        probs_data[:] = np.NaN
        raw_probs_data = np.empty(shape=(len(df),128))
        raw_probs_data[:] = np.NaN
        
        # Arguments
        args = parse_generate_args()
        dir_weights = dir_base + 'Code/MusicAnalysis/MusicModels/MT/TrainedModels/PretrainMaestroE150Fine/weights/'
        fn_weights = 'epoch_0023.pickle'
        args.model_weights = dir_weights + fn_weights
        args.rpr            = True
        args.max_sequence   = 2048
        args.primer_len     = len_context
               
        # Loop over compositions
        i_row = 0
        for fn in filenames:
            
            # Encode midi data of current composition
            args.primer_file = dir_midi + fn
            raw_mid = encode_midi(args.primer_file)
            
            # loop over notes (note onset events only)
            note_positions = [i for i, j in enumerate(raw_mid) if j <= 127] # find note onset events
        
            for i_note, note_pos in enumerate(note_positions):
                    
                    if i_note < args.primer_len:
                        sequence_start = 0
                        args.num_prime = note_pos
                    else:
                        sequence_start = note_positions[i_note - args.primer_len]
                        args.num_prime = note_pos - sequence_start
                        #raw_mid_interim = raw_mid[i_note-args.primer_len:]
                        #primer, _  = process_midi(raw_mid_interim, args.num_prime, random_seq=False)
                    
                    primer, _  = process_midi(raw_mid, args.num_prime, sequence_start, random_seq=False)
                    primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())
            
                    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                                             d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                                             max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())
            
                    model.load_state_dict(torch.load(args.model_weights,map_location=torch.device('cpu')))
                    
                    # Length of target sequence
                    args.target_seq_length =  args.num_prime + 1
                    
                    # random sampling generation
                    model.eval()
                    with torch.set_grad_enabled(False):
                        probs, raw_probs = model.get_probs(primer, i_note, target_seq_length = args.target_seq_length, raw_mid = raw_mid)
                    
                    probs_data[i_row, :] = probs
                    raw_probs_data[i_row, :] = raw_probs
        
                    print('Event ' + str(df.loc[i_row, 'note_number'] + 1) + ' / ' + str(df[df['filenumber'] == df.loc[i_row, 'filenumber']]['note_number'].max() + 1) + ' in composition ' + str(df.loc[i_row, 'filenumber'] + 1) + ' / ' + str(len(filenames)))
                    i_row += 1
            
            #Normalize probabilites to sum up to 1
            probs_data = norm_neg_log_probs(probs_data)
        
            # Assume uniform distribution for the first note in each composition
            probs_data[df['note_number']==0] = -np.log(np.repeat(1/128, 128))
            
            np.savetxt(dir_out + fn_out, probs_data, delimiter=",")
            np.savetxt(dir_out + fn_out_raw, raw_probs_data, delimiter=",")