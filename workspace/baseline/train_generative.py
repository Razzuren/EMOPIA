import os
import json
import argparse
import numpy      as np
import tensorflow as tf
import midi_encoder as me
from metrics import MetricsLogger

# Directory where the checkpoints will be saved
TRAIN_DIR = "./trained"
seq_length = 0

def generative_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def build_generative_model(vocab_size, embed_dim, lstm_units, lstm_layers, batch_size, dropout=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(seq_length,), batch_size=batch_size))

    model.add(tf.keras.layers.Embedding(vocab_size, embed_dim))

    for i in range(max(1, lstm_layers)):
        model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=True, stateful=True, dropout=dropout, recurrent_dropout=dropout))

    model.add(tf.keras.layers.Dense(vocab_size))
    print(model.summary())

    return model

def build_char2idx(train_vocab, test_vocab):
    # Merge train and test vocabulary
    vocab = list(train_vocab | test_vocab)
    vocab.sort()

    # Calculate vocab size
    vocab_size = len(vocab)

    # Create dict to support char to index conversion
    char2idx = { char:i for i,char in enumerate(vocab) }

    # Save char2idx encoding as a json file for generate midi later
    with open(os.path.join(TRAIN_DIR, "char2idx.json"), "w") as f:
        json.dump(char2idx, f)

    return char2idx, vocab_size


def build_dataset(text, char2idx, seq_length, batch_size, buffer_size=10000):
    if not text.strip():
        print("Erro: O texto está vazio!")
        return tf.data.Dataset.from_tensor_slices([])

    text_as_int = np.array([char2idx[c] for c in text.split() if c.strip() != ""])

    if len(text_as_int) == 0:
        print("Erro: Nenhum token foi convertido para índice!")
        return tf.data.Dataset.from_tensor_slices([])

    print(f"Tamanho do array text_as_int: {len(text_as_int)}")

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(__split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset


def train_generative_model(model, train_dataset, test_dataset, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=generative_loss)

    checkpoint_prefix = os.path.join(TRAIN_DIR, "generative_ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    metrics_logger = MetricsLogger(model, learning_rate,
                                   log_file=os.path.join(TRAIN_DIR, "training_metrics.csv"))

    return model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[checkpoint_callback, metrics_logger]
    )

def __split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=2, help="LSTM layers.")
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs.")
    parser.add_argument('--seqlen', type=int, default=100, help="Sequence lenght.")
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--drop', type=float, default=0.0, help="Dropout.")
    opt = parser.parse_args()

    seq_length = opt.seqlen
    # Encode midi files as text with vocab
    train_text, train_vocab = me.load(opt.train)
    test_text, test_vocab = me.load(opt.test)

    # Build dictionary to map from char to integers
    char2idx, vocab_size = build_char2idx(train_vocab, test_vocab)

    # Build dataset from encoded unlabelled midis
    train_dataset = build_dataset(train_text, char2idx, opt.seqlen, opt.batch)
    test_dataset = build_dataset(test_text, char2idx, opt.seqlen, opt.batch)

    # Build generative model
    generative_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, opt.batch, opt.drop)

    if opt.model:
        # If pre-trained model was given as argument, load weights from disk
        print("Loading weights...")
        generative_model.load_weights("trained/generative_ckpt_10.weights.h5")

    # Train model
    history = train_generative_model(generative_model, train_dataset, test_dataset, opt.epochs, opt.lrate)