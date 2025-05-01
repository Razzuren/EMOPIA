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

import midi_encoder as me
from train_generative import build_generative_model
from train_classifier import preprocess_sentence

###########################################
# Conversão de tokens -> durações, eventos
###########################################

def parse_duration_token(token):
    """
    Ex.: 'd_quarter_1' => type='quarter', dots=1 => quarterLength=1.5
         'd_16th_0'    => type='16th',  dots=0 => quarterLength=0.25
    """
    parts = token.split("_")
    if len(parts) < 3:
        return 0.25

    dur_type = parts[1]
    try:
        dots = int(parts[2])
    except:
        dots = 0

    d = m21.duration.Duration()
    d.type = dur_type
    d.dots = dots
    return d.quarterLength

def tokens_to_events(tokens):
    """
    Converte lista de tokens em lista de eventos (note_on, note_off).
    """
    events = []
    qn_offset = 0.0
    current_duration = 0.25
    current_velocity = 64
    bpm = 65  # fixo; se quiser dinâmica, implemente "t_"

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token.startswith("w_"):
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

        elif token.startswith("n_"):
            pitch_str = token.split("_")[1]
            try:
                pitch = int(pitch_str)
            except:
                pitch = 60

            start_sec = qn_offset * (60.0 / bpm)
            end_sec   = (qn_offset + current_duration) * (60.0 / bpm)

            events.append(("note_on", pitch, current_velocity, start_sec))
            events.append(("note_off", pitch, current_velocity, end_sec))

        # Se quiser suportar "t_", etc., implemente

    # Ordena eventos pelo tempo
    events.sort(key=lambda e: e[3])
    return events

def play_events_realtime(events):
    """
    Toca os eventos em tempo real (time.sleep).
    """
    if not events:
        return

    port_name = "ehocarai 2"  # Ajuste conforme mido.get_output_names()
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

def sample_next(predictions, k=3):
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values  = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < 0.5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.nn.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], p=p_choices)
    return predicted_id

def generate_in_batches_thread(model, char2idx, idx2char, init_text,
                               batch_size, total_steps, k, out_queue,
                               stop_event,
                               collector_queue=None):
    """
    Gera tokens em lotes e coloca cada lote na fila 'out_queue' para ser tocado.
    Também pode armazenar os tokens num 'collector_queue' para salvar depois.
    """
    init_text = preprocess_sentence(init_text)
    predictions = None

    # "aquece" o modelo
    for c in init_text.split():
        if c in char2idx:
            inp = tf.expand_dims([char2idx[c]], 0)
            predictions = model(inp)

    steps_so_far = 0
    # Se quisermos registrar também os tokens iniciais
    if collector_queue and init_text.strip():
        collector_queue.put(init_text.split())

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
            # Envia tokens para a Tocadora
            out_queue.put(block_tokens)

            # Se quisermos salvar, enviamos também ao collector
            if collector_queue:
                collector_queue.put(block_tokens)

    # Ao final, sinaliza fim para Tocadora
    out_queue.put("END")
    print("[Geradora] Fim da geração. Inseriu 'END' na fila.")

    # E também sinaliza no collector
    if collector_queue:
        collector_queue.put("END")


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

        events = tokens_to_events(block)
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
    opt = parser.parse_args()

    # Carrega char2idx
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)
    idx2char = {v: k for k, v in char2idx.items()}

    # Carrega modelo
    vocab_size = len(char2idx)
    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(opt.model)
    model.build(tf.TensorShape([1, 0]))

    # Pasta de saída
    GENERATED_DIR = "./generated"
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # Marca tempo inicial
    start_time = time.time()

    # Cria fila e event
    q_play = queue.Queue()        # Para enviar tokens -> Tocadora
    q_collect = queue.Queue()     # Para coletar todos os tokens e salvar depois
    stop_event = threading.Event()

    # Cria threads
    t_gen = threading.Thread(
        target=generate_in_batches_thread,
        args=(model, char2idx, idx2char,
              opt.seqinit, opt.batchsize, opt.totalsteps, opt.k,
              q_play, stop_event, q_collect),
        daemon=True
    )

    t_play = threading.Thread(
        target=play_thread,
        args=(q_play, stop_event),
        daemon=True
    )

    # Inicia threads
    t_gen.start()
    t_play.start()

    # Espera concluir
    t_gen.join()
    t_play.join()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / opt.totalsteps

    print("=== Fim da execução ===")

    # Agora, vamos recolher todos os tokens do q_collect e montar um único texto
    all_tokens = []
    while not q_collect.empty():
        item = q_collect.get()
        if item == "END":
            # Sinal de que a geradora acabou
            continue
        elif isinstance(item, list):
            # São tokens
            all_tokens.extend(item)
        elif isinstance(item, str):
            # Pode ser init_text
            all_tokens.extend(item.split())

    # Converte para string
    final_text = " ".join(all_tokens)
    print("Total de tokens coletados:", len(all_tokens))

    tokens = final_text.split()
    found_first_t = False
    filtered_tokens = []

    for tok in tokens:
        if tok.startswith("t_"):
            if not found_first_t:
                filtered_tokens.append(tok)  # mantém o primeiro t_
                found_first_t = True
            # senão descarta
        else:
            filtered_tokens.append(tok)

    final_text_filtered = " ".join(filtered_tokens)

    # Salva .mid
    midi_path = os.path.join(GENERATED_DIR, "generated_realtime.mid")
    me.write(final_text, midi_path)
    print(f"MIDI salvo em: {midi_path}")

    # Salva métricas
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M")
    metrics_filename = f"{timestamp}_generation_metrics.txt"
    metrics_path = os.path.join(GENERATED_DIR, metrics_filename)
    with open(metrics_path, "w") as f:
        f.write(f"Total generation time (seconds): {total_time}\n")
        f.write(f"Average time per token (seconds): {avg_time_per_token}\n")

    print(f"Métricas de geração salvas em: {metrics_path}")
