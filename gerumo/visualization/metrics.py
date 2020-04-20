from os.path import join
import matplotlib.pyplot as plt

def plot_model_training_history(history, training_time, model_name, epochs, output_folder="."):
    fig = plt.figure(figsize=(12,6))
    epochs = [i for i in range(1, epochs+1)]
    plt.plot(epochs, history.history['loss'], "*--", label="Train")
    plt.plot(epochs, history.history['val_loss'], "*--", label="Validation")
    plt.title(f'Model {model_name} Training Loss\n Training time {training_time} [min]')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(epochs, rotation=-90)
    plt.grid()
    fig.savefig(join(output_folder, f'{model_name} - Training Loss.png'))
    plt.close(fig)
