import datetime
import os
import json
import argparse
import time
import numpy as np
import tensorflow as tf
import threading
import queue

import mido
import music21 as m21

from train_generative import build_generative_model
from train_classifier import preprocess_sentence


###########################################
# Conversão de tokens -> durações, eventos
###########################################

def parse_duration_token(token):
    """
    Ex.: 'd_quarter_1' => type='quarter', dots=1 => quarterLength=1.5
         'd_16th_0'    => type='16th', dots=0 => quarterLength=0.25
    """
    parts = token.split("_")
    if len(parts) < 3:
        return 0.25  # fallback

    dur_type = parts[1]  # 'quarter'
    try:
        dots = int(parts[2])
    except:
        dots = 0

    d = m21.duration.Duration()
    d.type = dur_type
    d.dots = dots

    q_len = d.quarterLength
    return q_len


def tokens_to_events(tokens):
    """
    Converte lista de tokens em lista de eventos (note_on, note_off, tempo).
    """
    events = []
    qn_offset = 0.0
    current_duration = 0.25
    current_velocity = 64
    bpm = 65

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token.startswith("w_"):
            # w_X => avança X "ticks" no offset
            w_str = token.split("_")[1]
            try:
                w_val = float(w_str)
            except:
                w_val = 1.0
            qn_offset += w_val

        elif token.startswith("d_"):
            current_duration = parse_duration_token(token)

        elif token.startswith("v_"):
            vel_str = token.split("_")[1]
            try:
                current_velocity = int(vel_str)
            except:
                pass

        # elif token.startswith("t_"):
        #     tempo_str = token.split("_")[1]
        #     try:
        #         bpm = float(tempo_str)
        #     except:
        #         bpm = 120.0

        elif token.startswith("n_"):
            pitch_str = token.split("_")[1]
            try:
                pitch = int(pitch_str)
            except:
                pitch = 60

            start_sec = qn_offset * (60.0 / bpm)
            end_sec = (qn_offset + current_duration) * (60.0 / bpm)

            # note_on e note_off
            events.append(("note_on", pitch, current_velocity, start_sec))
            events.append(("note_off", pitch, current_velocity, end_sec))

        else:
            # ignora tokens desconhecidos
            pass

    # ordena por tempo
    events.sort(key=lambda e: e[3])
    return events


def play_events_realtime(events):
    """
    Reproduz a lista de eventos em tempo real, usando time.sleep()
    para sincronizar.
    """
    if not events:
        return

    # Ajuste o nome da porta conforme seu sistema
    port_name = "ehocarai 2"
    outport = mido.open_output(port_name)
    print(f"[Tocadora] Tocando {len(events)} eventos na porta '{port_name}'.")

    start_time = time.time()
    for ev_type, pitch, velocity, ev_sec in events:
        now = time.time() - start_time
        wait = ev_sec - now
        if wait > 0:
            time.sleep(wait)

        if ev_type == "note_on":
            msg = mido.Message('note_on', note=pitch, velocity=velocity)
            outport.send(msg)
        elif ev_type == "note_off":
            msg = mido.Message('note_off', note=pitch, velocity=0)
            outport.send(msg)

    outport.close()
    print("[Tocadora] Concluiu a execução deste bloco.")


####################################################
# Thread 1 (Geradora): gera tokens em lotes (batches)
####################################################

def sample_next(predictions, k):
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < 0.5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.nn.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], p=p_choices)
    return predicted_id


def generate_in_batches_thread(model, char2idx, idx2char, init_text,
                               batch_size, total_steps, k, out_queue,
                               stop_event):
    """
    Gera tokens em lotes e coloca cada lote na fila 'out_queue'.
    Quando terminar, coloca 'END' na fila.
    """
    init_text = preprocess_sentence(init_text)
    predictions = None

    # "Aquece" o modelo
    for c in init_text.split():
        if c in char2idx:
            inp = tf.expand_dims([char2idx[c]], 0)
            predictions = model(inp)

    steps_so_far = 0
    while steps_so_far < total_steps and not stop_event.is_set():
        block_size = min(batch_size, total_steps - steps_so_far)
        block_tokens = []

        for _ in range(block_size):
            if stop_event.is_set():
                break
            preds_np = tf.squeeze(predictions, 0).numpy()
            pred_id = sample_next(preds_np, k)
            token = idx2char[pred_id]
            block_tokens.append(token)

            inp = tf.expand_dims([pred_id], 0)
            predictions = model(inp)

        steps_so_far += block_size
        if block_tokens:
            out_queue.put(block_tokens)

    # Sinaliza fim
    out_queue.put("END")
    print("[Geradora] Fim da geração. Coloquei 'END' na fila.")


###################################################################
# Thread 2 (Tocadora): lê lotes da fila, converte em eventos e toca
###################################################################

def play_thread(in_queue, stop_event):
    """
    Lê lotes (listas de tokens) da fila 'in_queue'. Se receber "END", termina.
    Caso contrário, converte tokens->eventos e chama play_events_realtime().
    """
    while not stop_event.is_set():
        try:
            block = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if block == "END":
            print("[Tocadora] Recebeu 'END', encerrando.")
            break

        # converte tokens em eventos
        events = tokens_to_events(block)
        # toca em tempo real
        play_events_realtime(events)

    print("[Tocadora] Thread finalizada.")


##################
# Script principal
##################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='batch_realtime_generator_multithread')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ch2ix', type=str, required=True)
    parser.add_argument('--embed', type=int, required=True)
    parser.add_argument('--units', type=int, required=True)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--seqinit', type=str, default="t_120")
    parser.add_argument('--totalsteps', type=int, default=256)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--cellix', type=int, default=-2)
    parser.add_argument('--override', type=str, default="")
    opt = parser.parse_args()

    # Carrega char2idx
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)
    idx2char = {v: k for k, v in char2idx.items()}

    # Carrega modelo
    model = build_generative_model(len(char2idx), opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(opt.model)
    model.build(tf.TensorShape([1, 0]))

    # Filtra override se precisar (não implementado neste script, mas poderia)
    # ...

    # Cria fila e event
    q = queue.Queue()
    stop_event = threading.Event()

    # Cria threads
    t_gen = threading.Thread(
        target=generate_in_batches_thread,
        args=(model, char2idx, idx2char, opt.seqinit,
              opt.batchsize, opt.totalsteps, opt.k,
              q, stop_event),
        daemon=True
    )

    t_play = threading.Thread(
        target=play_thread,
        args=(q, stop_event),
        daemon=True
    )

    # Inicia threads
    t_gen.start()
    t_play.start()

    # Espera concluir
    t_gen.join()
    t_play.join()

    print("=== Fim da execução ===")
