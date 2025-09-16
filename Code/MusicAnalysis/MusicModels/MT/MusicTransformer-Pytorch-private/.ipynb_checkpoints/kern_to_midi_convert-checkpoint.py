import music21
import os
from pathlib import Path

input_dir = 'cudlun_kern_stimuli_with_probe'
output_dir = 'cudlun_midi_stimuli_with_probe'
os.makedirs(output_dir, exist_ok=True)

def clean_kern_text(text: str) -> str:
    cleaned_lines = []
    for ln in text.splitlines():
        s = ln.rstrip("\n\r")
        if s.strip() == '...':
            # remove placeholder lines
            continue
        if s.startswith('!!!') and ':' not in s:
            # demote malformed reference record to a regular comment
            s = '!! ' + s.lstrip('!')
        cleaned_lines.append(s)
    return "\n".join(cleaned_lines) + "\n"

file_counter = 0
for file in os.listdir(input_dir):
    if not file.endswith('.krn'):
        continue

    in_path = Path(input_dir, file)
    print(f"Processing {file}...")
    try:
        raw = in_path.read_text(encoding='utf-8', errors='replace')
        cleaned = clean_kern_text(raw)

        # parse from cleaned string (no temp file needed)
        score = music21.converter.parse(cleaned, format='humdrum')

        out_file = os.path.splitext(file)[0] + '.mid'
        out_path = Path(output_dir, out_file)
        score.write('midi', fp=str(out_path))
        file_counter += 1

    except Exception as e:
        print(f"SKIPPED: {file} ({e})")

print("The number of files successfully converted from kern to midi is", file_counter)
