import os
import pickle
import third_party.midi_processor.processor as midi_processor
import random
from math import ceil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess a folder of generic MIDI files into split .pickle datasets for Music Transformer.")
    parser.add_argument("-input_dir", type=str, required=True, help="Directory containing raw .mid files")
    parser.add_argument("-output_dir", type=str, required=True, help="Directory to save preprocessed train/val/test data")
    parser.add_argument("-train_ratio", type=float, default=0.8, help="Ratio of files to put in train split")
    parser.add_argument("-val_ratio", type=float, default=0.0, help="Ratio of files to put in val split")
    parser.add_argument("-test_ratio", type=float, default=0.2, help="Ratio of files to put in test split")
    parser.add_argument("-seed", type=int, default=2021, help="Random seed for splitting")
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    files.sort()

    dir_train = os.path.join(args.output_dir, "train")
    dir_val = os.path.join(args.output_dir, "val")
    dir_test = os.path.join(args.output_dir, "test")
    
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_val, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    parts = {}
    if args.train_ratio > 0: parts['train'] = args.train_ratio
    if args.val_ratio > 0: parts['val'] = args.val_ratio
    if args.test_ratio > 0: parts['test'] = args.test_ratio

    split = []
    for part in parts.keys():
        num_part = ceil((parts[part] * len(files)))
        split.extend([part] * num_part)
    split = split[:len(files)]   
    random.seed(args.seed)
    random.shuffle(split)

    count_total, count_train, count_val, count_test = 0, 0, 0, 0
    skipped_files = []

    for i, fn_in in enumerate(files):
        f_in = os.path.join(args.input_dir, fn_in)
        fn_out = fn_in + ".pickle"
        
        if split[i] == 'train':
            o_file = os.path.join(dir_train, fn_out)
            count_train += 1
        elif split[i] == 'val':
            o_file = os.path.join(dir_val, fn_out)
            count_val += 1
        elif split[i] == 'test':
            o_file = os.path.join(dir_test, fn_out)
            count_test += 1

        try:
            prepped = midi_processor.encode_midi(f_in)
            with open(o_file, "wb") as o_stream:
                pickle.dump(prepped, o_stream)
            count_total += 1
            if count_total % 50 == 0:
                print(f"{count_total}/{len(files)}")
        except Exception as e:
            print(f"SKIPPED: {fn_in} (reason: {e})")
            skipped_files.append(fn_in)

    print(f"Num Train: {count_train}, Num Val: {count_val}, Num Test: {count_test}")
    print(f"Successfully Preprocessed: {count_total}, Skipped: {len(skipped_files)}")

if __name__ == "__main__":
    main()
