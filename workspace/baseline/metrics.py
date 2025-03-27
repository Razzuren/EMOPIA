import os
import csv
import tensorflow as tf

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, learning_rate, log_file="training_metrics.csv"):
        super().__init__()
        self.learning_rate = learning_rate
        self.log_file = log_file

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "val_loss",
                                 "learning_rate",
                                 "num_lstm_layers", "lstm_units", "dropouts"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        lstm_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        num_lstm_layers = len(lstm_layers)
        lstm_units = [l.units for l in lstm_layers]
        dropouts = [l.dropout for l in lstm_layers]

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                loss,
                val_loss,
                self.learning_rate,
                num_lstm_layers,
                str(lstm_units),   # Ex.: "[512, 512]" se tiver 2 camadas
                str(dropouts)      # Ex.: "[0.0, 0.0]" se dropout=0
            ])
