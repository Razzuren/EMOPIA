"""
realtime_allinone.py  –  geração e execução MIDI em tempo‑real com o
Transformer EMOPIA.

Fluxo completo num único arquivo:
    MIDI → corpus‑dict → events → tokens  (in‑memory)
    tokens (prompt) aquecem o Transformer → geração incremental
    blocos convertidos em eventos → enviados via porta MIDI
    no final, peça completa salva como .mid

Requisitos:
    pip install miditoolkit mido python-rtmidi numpy torch fast_transformers
"""

# ────────────────────────── IMPORTS ──────────────────────────────
import os, time, json, queue, threading, argparse, random, string
from collections import OrderedDict, defaultdict

import numpy as np
import mido
import torch
import miditoolkit
import pickle
from miditoolkit.midi.containers import Note, Instrument, TempoChange, Marker

from models import TransformerModel  # seu models.py
from utils import sampling, write_midi  # util do repo

# ───────────────────── CONSTANTES & BINS ────────────────────────
PORT_NAME = "loopMIDI 1"  # <- altere para sua porta MIDI
BATCH_SIZE = 32
TOTAL_TOK = 512
FIXED_BPM = 120

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

# bins originalmente usados pelo dataset
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 64 + 1, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=int)
DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 8, BEAT_RESOL * 8 + 1, BEAT_RESOL / 8)

MIN_VELOCITY = 40
NOTE_SORTING = 1  # 0: asc, 1: desc
INSTR_NAME_MAP = {"piano": 0}


def _quant_round(x, base):
    return int(np.round(x / base) * base)


# ──────────────────── 1. MIDI → corpus‑dict ─────────────────────

def midi_to_corpus_dict(midi: miditoolkit.midi.parser.MidiFile, midi_path):
    """Extrai notas, acordes, tempos, emoção + metadados (em memória)."""

    # emoção vem dos 2 primeiros chars do filename, se existir
    emo_tag = 'Q1'
    if midi:
        emo_tag = os.path.basename(midi_path)[:2]

    # coleta notas track piano
    instr_notes = defaultdict(list)
    for instr in midi.instruments:
        if instr.name not in INSTR_NAME_MAP:
            continue
        idx = INSTR_NAME_MAP[instr.name]
        for n in instr.notes:
            instr_notes[idx].append(n)
        instr_notes[idx].sort(key=lambda n: (n.start, -n.pitch) if NOTE_SORTING else (n.start, n.pitch))

    # acordes / tempos / markers
    chords = [m for m in midi.markers if 'Boundary' not in m.text and not m.text.startswith('global')]
    chords.sort(key=lambda m: m.time)
    tempos = sorted(midi.tempo_changes, key=lambda t: t.time)

    # offset vazio inicial
    first_start = min(n.start for notes in instr_notes.values() for n in notes)
    offset_bar = _quant_round(first_start, TICK_RESOL) // BAR_RESOL
    last_bar = int(np.ceil(max(n.start for notes in instr_notes.values() for n in notes) / BAR_RESOL)) - offset_bar

    # grids
    note_grid = defaultdict(lambda: defaultdict(list))  # instr -> time -> [Note]
    for idx, notes in instr_notes.items():
        for n in notes:
            n.start -= offset_bar * BAR_RESOL
            n.end -= offset_bar * BAR_RESOL
            q_start = _quant_round(n.start, TICK_RESOL)

            # velocity & duration bins
            n.velocity = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS - n.velocity))]
            n.velocity = max(MIN_VELOCITY, n.velocity)
            dur_ticks = _quant_round(n.end - n.start, TICK_RESOL)
            dur_ticks = min(dur_ticks, BAR_RESOL)
            n.end = n.start + dur_ticks
            note_grid[idx][q_start].append(n)

    chord_grid = defaultdict(list)
    for c in chords:
        c.time = max(0, c.time - offset_bar * BAR_RESOL)
        chord_grid[_quant_round(c.time, TICK_RESOL)].append(c)

    tempo_grid = defaultdict(list)
    for t in tempos:
        t.time = max(0, t.time - offset_bar * BAR_RESOL)
        t.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - t.tempo))]
        tempo_grid[_quant_round(t.time, TICK_RESOL)].append(t)

    corpus = {
        'notes': note_grid,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'metadata': {
            'last_bar': last_bar,
            'emotion': emo_tag
        }
    }
    return corpus


# ───────────── 2. corpus‑dict → events‑list (8‑field dict) ─────

def corpus_to_events(corpus):
    """Replica lógica do script corpus2event_cp para um dicionário em RAM."""
    emo_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    events = []

    def emo_event(tag):
        e = {k: 0 for k in ('tempo', 'chord', 'bar-beat', 'type', 'pitch', 'duration', 'velocity', 'emotion')}
        e['emotion'] = tag;
        e['type'] = 'Emotion';
        return e

    def bar_event():
        e = {k: 0 for k in ('tempo', 'chord', 'bar-beat', 'type', 'pitch', 'duration', 'velocity', 'emotion')}
        e['bar-beat'] = 'Bar';
        e['type'] = 'Metrical';
        return e

    def metr_event(tempo, chord, pos):
        e = {k: 0 for k in ('tempo', 'chord', 'bar-beat', 'type', 'pitch', 'duration', 'velocity', 'emotion')}
        e['tempo'] = tempo;
        e['chord'] = chord;
        e['bar-beat'] = pos;
        e['type'] = 'Metrical';
        return e

    def note_event(pitch, dur, vel):
        e = {k: 0 for k in ('tempo', 'chord', 'bar-beat', 'type', 'pitch', 'duration', 'velocity', 'emotion')}
        e['pitch'] = pitch;
        e['duration'] = dur;
        e['velocity'] = vel;
        e['type'] = 'Note';
        return e

    last_bar = corpus['metadata']['last_bar']
    events.append(emo_event(emo_map.get(corpus['metadata']['emotion'], 1)))

    for bar in range(last_bar):
        base_time = bar * BAR_RESOL
        events.append(bar_event())
        for step in range(0, BAR_RESOL, TICK_RESOL):
            t = base_time + step
            pos_txt = f"Beat_{step // TICK_RESOL}"
            pos_ev = [];
            have_met = False

            # tempo / chord
            tempo_list = corpus['tempos'][t];
            chord_list = corpus['chords'][t]
            if tempo_list or chord_list:
                tempo_txt = f"Tempo_{tempo_list[-1].tempo}" if tempo_list else 'CONTI'
                chord_txt = chord_list[-1].text.split('_')[0] + "_" + chord_list[-1].text.split('_')[
                    1] if chord_list else 'CONTI'
                pos_ev.append(metr_event(tempo_txt, chord_txt, pos_txt));
                have_met = True

            # notes
            notes = corpus['notes'][0][t]
            if notes:
                if not have_met:
                    pos_ev.append(metr_event('CONTI', 'CONTI', pos_txt))
                for n in notes:
                    pos_ev.append(note_event(f"Note_Pitch_{n.pitch}", f"Note_Duration_{n.duration}",
                                             f"Note_Velocity_{n.velocity}"))
            events.extend(pos_ev)
    events.append(bar_event())  # bar final
    events.append(
        {'type': 'EOS', **{k: 0 for k in ('tempo', 'chord', 'bar-beat', 'pitch', 'duration', 'velocity', 'emotion')}})
    return events


# ───────────── 3. events‑list → tokens  usando dict ------------

def events_to_tokens(events, event2word):
    class_keys = list(event2word.keys())
    return np.array([[event2word[k][ev[k]] for k in class_keys] for ev in events], dtype=np.int64)


# ───────────— conversão direta MIDI → prompt tokens —───────────

def midi_to_prompt_tokens(midi_path, dictionary):
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    corpus = midi_to_corpus_dict(midi, midi_path)
    events = corpus_to_events(corpus)
    return events_to_tokens(events, dictionary[0])


# ─────────────────────── Player & Threads ───────────────────────

def tokens_to_events(block, w2e, bpm=FIXED_BPM):
    evs, qn = [], 0.0
    cur_vel, cur_dur = 64, 0.25
    wt, wbb, wp, wd, wv = (w2e['type'], w2e['bar-beat'], w2e['pitch'], w2e['duration'], w2e['velocity'])
    for row in block:
        typ = wt[int(row[3])]
        if typ == 'Metrical':
            bb = wbb[int(row[2])]
            if bb == 'Bar':
                qn = int(qn // 1) + 1
            elif bb.startswith('Beat_'):
                qn = int(bb.split('_')[1]) * 0.25
        elif typ == 'Note':
            pitch = int(wp[int(row[4])].split('_')[-1])
            cur_dur = int(wd[int(row[5])].split('_')[-1]) / 960
            cur_vel = int(wv[int(row[6])].split('_')[-1])
            start = qn * 60 / bpm
            end = (qn + cur_dur) * 60 / bpm
            evs.append(('note_on', pitch, cur_vel, start))
            evs.append(('note_off', pitch, 0, end))
    evs.sort(key=lambda e: e[3]);
    return evs


def play_events(events, port):
    if not events: return
    t0 = time.time()
    for typ, p, v, t_sec in events:
        wait = t_sec - (time.time() - t0);
        if wait > 0: time.sleep(wait)
        port.send(mido.Message(typ, note=p, velocity=v))


def generator_thread(net, prompt_tokens, q_out, stop_evt):
    device = next(net.parameters()).device
    with torch.no_grad():
        h = y_type = memory = None
        for tok in prompt_tokens:
            inp = torch.tensor(tok).long()[None, None].to(device)
            h, y_type, memory = net.forward_hidden(inp, memory, is_training=False)
        buf = [];
        steps = 0
        while steps < TOTAL_TOK and not stop_evt.is_set():
            nxt, _ = net.froward_output_sampling(h, y_type)
            if nxt is None: continue
            buf.append(nxt);
            steps += 1
            inp = torch.from_numpy(nxt).long()[None, None].to(device)
            h, y_type, memory = net.forward_hidden(inp, memory, is_training=False)
            if len(buf) == BATCH_SIZE:
                q_out.put(np.stack(buf));
                buf = []
        if buf: q_out.put(np.stack(buf))
        q_out.put("END");
        stop_evt.set()


def player_thread(q_in, dictionary, port_name, stop_evt):
    _, w2e = dictionary;
    port = mido.open_output(port_name)
    while not stop_evt.is_set():
        blk = q_in.get()
        if isinstance(blk, str) and blk == "END": break
        play_events(tokens_to_events(blk, w2e), port)
    port.close()


# ───────────────────────── main func ───────────────────────────

def realtime_generate(ckpt_dir, loss_tag, dict_path, prompt_midi=None, prompt_tokens=None, port=PORT_NAME):
    dictionary = pickle.load(open(dict_path, 'rb'))
    n_class = [len(dictionary[0][k]) for k in dictionary[0]]

    net = TransformerModel(n_class, is_training=False)
    net.cuda().eval()
    state = torch.load(os.path.join(ckpt_dir, f"loss_{loss_tag}_params.pt"))
    try:
        net.load_state_dict(state)
    except:
        net.load_state_dict(OrderedDict((k[7:], v) for k, v in state.items()))

    if prompt_midi:
        prompt_tok = midi_to_prompt_tokens(prompt_midi, dictionary)
    elif prompt_tokens:
        prompt_tok = np.load(prompt_tokens, allow_pickle=True)
    else:
        raise ValueError('Forneça --prompt_midi ou --prompt_tokens')

    q = queue.Queue(maxsize=8);
    stop_evt = threading.Event()
    th_gen = threading.Thread(target=generator_thread, args=(net, prompt_tok, q, stop_evt))
    th_play = threading.Thread(target=player_thread, args=(q, dictionary, port, stop_evt))
    th_gen.start();
    th_play.start();
    th_gen.join();
    th_play.join()

    # salva peça gerada
    toks = []
    while not q.empty():
        item = q.get_nowait()
        if not isinstance(item, str): toks.append(item)
    if toks:
        toks = np.concatenate(toks, axis=0)
        out_mid = f"gen_{''.join(random.choice(string.ascii_lowercase) for _ in range(6))}.mid"
        write_midi(toks, out_mid, dictionary[1])
        print('>> MIDI salvo em', out_mid)


# ─────────────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Geração em tempo‑real com Transformer EMOPIA')
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--loss', required=True)
    ap.add_argument('--dict', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prompt_midi')
    g.add_argument('--prompt_tokens')
    ap.add_argument('--port', default=PORT_NAME)
    ap.add_argument('--batch', type=int, default=BATCH_SIZE)
    ap.add_argument('--total', type=int, default=TOTAL_TOK)
    args = ap.parse_args()

    # override consts if user set
    globals()['BATCH_SIZE'] = args.batch
    globals()['TOTAL_TOK'] = args.total
    globals()['PORT_NAME'] = args.port

    realtime_generate(args.ckpt_dir, args.loss, args.dict,
                      prompt_midi=args.prompt_midi,
                      prompt_tokens=args.prompt_tokens,
                      port=args.port)
