import os
import json
import argparse
import numpy as np
import tensorflow as tf
import midi_encoder as me

from train_generative import build_generative_model
from train_classifier import preprocess_sentence

GENERATED_DIR = './generated'

def override_neurons(model, layer_idx, override):
    h_state, c_state = model.get_layer(index=layer_idx).states

    c_state = c_state.numpy()
    for neuron, value in override.items():
        c_state[:, int(neuron)] = int(value)

    model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))

def sample_next(predictions, k):
    # Sample usando distribuição categórica sobre o top k
    top_k = tf.math.top_k(predictions, k)
    top_k_choices = top_k[1].numpy().squeeze()
    top_k_values = top_k[0].numpy().squeeze()

    if np.random.uniform(0, 1) < 0.5:
        predicted_id = top_k_choices[0]
    else:
        p_choices = tf.math.softmax(top_k_values[1:]).numpy()
        predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]

    return predicted_id

def process_init_text(model, init_text, char2idx, layer_idx, override):
    # Pode chamar model.reset_states() se precisar zerar o estado
    print(init_text)
    predictions = None

    for c in init_text.split(" "):
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)
            # Se quiser modificar alguma neuron override, chame aqui
            # override_neurons(model, layer_idx, override)
            predictions = model(input_eval)
        except KeyError as b:
            if c != "":
                print("Can't process char", c, "because", b)

    return predictions

def remove_large_w_tokens(generated_tokens, threshold=200):
    """Remove tokens que comecem com 'w_' e cujo número seja maior que 'threshold'."""
    filtered_tokens = []
    for token in generated_tokens:
        if token.startswith("w_"):
            try:
                number_part = int(token.split("_")[1])
                if number_part <= threshold:
                    filtered_tokens.append(token)
            except ValueError:
                # Não conseguiu converter, mantém por default
                filtered_tokens.append(token)
        else:
            filtered_tokens.append(token)
    return filtered_tokens

def remove_t_tokens(generated_tokens):
    """Remove tokens que começam com 't_'."""
    return [token for token in generated_tokens if not token.startswith('t_')]

def generate_midi(model, char2idx, idx2char, init_text="", seq_len=256, k=3, layer_idx=-2, override={}):
    # Ajusta init_text (ex.: adiciona tokens de pad, etc., se necessário)
    init_text = preprocess_sentence(init_text)

    # Processa texto inicial
    predictions = process_init_text(model, init_text, char2idx, layer_idx, override)

    # Cria lista de tokens gerados
    midi_generated = []

    for i in range(seq_len):
        predictions = tf.squeeze(predictions, 0).numpy()
        predicted_id = sample_next(predictions, k)
        midi_generated.append(idx2char[predicted_id])

        input_eval = tf.expand_dims([predicted_id], 0)
        predictions = model(input_eval)

    # Primeiro, remove tokens w_ acima do threshold
    midi_generated = remove_large_w_tokens(midi_generated, threshold=130)

    # Depois, remove tokens t_
    midi_generated = remove_t_tokens(midi_generated)

    # Retorna o texto final
    return init_text + " " + " ".join(midi_generated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="t_120", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.")
    parser.add_argument('--cellix', type=int, default=-2, help="LSTM layer to use as encoder.")
    parser.add_argument('--override', type=str, default="", help="JSON file with neuron values to override.")
    opt = parser.parse_args()

    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Tenta carregar overrides
    override = {}
    try:
        with open(opt.override) as f:
            override = json.load(f)
    except FileNotFoundError:
        print("Override JSON file not provided.")

    # idx2char invertido
    idx2char = {idx:char for char,idx in char2idx.items()}

    vocab_size = len(char2idx)

    model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    model.load_weights(opt.model)
    model.build(tf.TensorShape([1,0]))

    # Gera
    midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, layer_idx=opt.cellix, override=override)
    print("Tokens Gerados:", midi_txt)

    # Escreve arquivo .mid
    out_path = os.path.join(GENERATED_DIR, "generated.mid")
    me.write(midi_txt, out_path)
    print(f"MIDI salvo em: {out_path}")
