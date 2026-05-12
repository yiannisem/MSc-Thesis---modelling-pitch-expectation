import os
import re
import json
import mido

def tokenize_lisp(text):
    # A regex to match strings, parentheses, and keywords/numbers
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|\(|\)|[^\s()]+', text)
    return tokens

def parse_lisp(tokens):
    stack = [[]]
    for token in tokens:
        if token == '(':
            new_list = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif token == ')':
            if len(stack) > 1:
                stack.pop()
        else:
            if token.startswith('"') and token.endswith('"'):
                stack[-1].append(token[1:-1])
            else:
                try:
                    stack[-1].append(int(token))
                except ValueError:
                    try:
                        stack[-1].append(float(token))
                    except ValueError:
                        stack[-1].append(token)
    return stack[0]

def extract_songs_from_parsed(parsed_data):
    songs = {}
    def walk(node):
        if not isinstance(node, list):
            return
        if len(node) >= 2 and isinstance(node[0], str):
            # Check if this node looks like a song list: ['song_name', [note1], [note2], ...]
            # Note1 should be a list like [[':ONSET', 0], [':DELTAST', 0]...]
            if isinstance(node[1], list) and len(node[1]) > 0 and isinstance(node[1][0], list) and len(node[1][0]) == 2 and node[1][0][0] == ':ONSET':
                song_name = node[0]
                notes = node[1:] # the rest are notes
                songs[song_name] = notes
                return
        for child in node:
            if isinstance(child, list):
                walk(child)

    walk(parsed_data)
    return songs

def dictify_note(note_list):
    return {k: v for k, v in note_list}

def create_midi(song_notes, ticks_per_beat=24):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    events = []
    for raw_note in song_notes:
        n = dictify_note(raw_note)
        pitch = n.get(':CPITCH')
        onset = n.get(':ONSET')
        dur = n.get(':DUR')
        
        if pitch is None or onset is None or dur is None:
            continue
            
        events.append({'type': 'note_on', 'time': int(onset), 'note': int(pitch), 'velocity': 64})
        events.append({'type': 'note_off', 'time': int(onset + dur), 'note': int(pitch), 'velocity': 64})
        
    events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'note_off' else 1))
    
    last_time = 0
    for e in events:
        delta = e['time'] - last_time
        track.append(mido.Message(e['type'], note=e['note'], velocity=e['velocity'], time=delta))
        last_time = e['time']
        
    return mid

if __name__ == "__main__":
    base_folder = r"c:\DriveSync\Queen_Mary\Modules\Project\Kern_Music_Transformer"
    lisp_dir = os.path.join(base_folder, r"Data\MusicData\IDyOM training data")
    output_dir = os.path.join(base_folder, r"Data\MusicData\IDyOM_data_MIDI")
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_songs = 0
    
    for filename in os.listdir(lisp_dir):
        if not filename.endswith('.lisp'):
            continue
            
        print(f"Processing {filename}...")
        lisp_file = os.path.join(lisp_dir, filename)
        
        with open(lisp_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        tokens = tokenize_lisp(text)
        parsed = parse_lisp(tokens)
        songs = extract_songs_from_parsed(parsed)
        
        print(f"  Found {len(songs)} songs in {filename}.")
        total_songs += len(songs)
        
        for song_name, notes in songs.items():
            mid = create_midi(notes)
            # Make sure song_name is safe for filenames
            safe_name = "".join([c if c.isalnum() or c in ['-', '_'] else '_' for c in song_name])
            mid.save(os.path.join(output_dir, f"{safe_name}.mid"))
            
    print(f"Finished parsing. Generated {total_songs} MIDI files in '{output_dir}'.")
