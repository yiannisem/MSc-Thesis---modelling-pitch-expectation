import os
import pickle
import third_party.midi_processor.processor as midi_processor
import random
from math import ceil

# Directories
dir_cur     = os.path.dirname(__file__)
dir_base    = dir_cur[:dir_cur.find('Code')]
dir_mccc    = dir_base + 'Data/MusicData/MCCC_midi'
files       = os.listdir(dir_mccc)
files.sort()
dir_out     = dir_base + 'Data/MusicData/MCCC_midi_preproc'

dir_train = os.path.join(dir_out, "train")
os.makedirs(dir_train, exist_ok=True)
dir_val = os.path.join(dir_out, "val")
os.makedirs(dir_val, exist_ok=True)
dir_test = os.path.join(dir_out, "test")
os.makedirs(dir_test, exist_ok=True)

# Intialize count variables
count_total = 0
count_train = 0
count_val   = 0
count_test  = 0
i           = 0

# Partitioning the corpus
parts  = {'test': 0.2, 'train': 0.8}

split = []
for part in parts.keys():
    num_part = ceil((parts[part] * len(files)))
    split.extend([part] * num_part)
split = split[:len(files)]   
random.seed(2021)
random.shuffle(split)

skipped_files = []

for fn_in in files:
    f_in = os.path.join(dir_mccc, fn_in)
    if not os.path.isfile(f_in):
        continue
    fn_out = os.path.basename(f_in) + ".pickle"

    if(split[i] == 'train'):
        o_file = os.path.join(dir_train, fn_out)
        count_train += 1
    elif(split[i] == 'val'):
        o_file = os.path.join(dir_val, fn_out)
        count_val += 1
    elif(split[i] == 'test'):
        o_file = os.path.join(dir_test, fn_out)
        count_test += 1
    else:
        print("ERROR: Unrecognized split type:", split)
        continue

    try:
        prepped = midi_processor.encode_midi(f_in)
        with open(o_file, "wb") as o_stream:
            pickle.dump(prepped, o_stream)
        count_total += 1
        if(count_total % 50 == 0):
            print(count_total, "/", len(files))
    except Exception as e:
        print(f"SKIPPED: {fn_in} (reason: {e})")
        skipped_files.append(fn_in)

    i += 1

print("Num Train:", count_train)
print("Num Val:", count_val)
print("Num Test:", count_test)
print("Num Successfully Preprocessed:", count_total)
print("Num Skipped:", len(skipped_files))
print("Skipped files:", skipped_files)

