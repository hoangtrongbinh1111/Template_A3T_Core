import tensorflow as tf
from tool import get_model, get_sample_data
import os
import pickle
import numpy as np

int2label = {0: "normal", 1: "phishing"}


async def infer(url_sample, labId, ckpt_number, model_type, embed_size, num_neurons, num_layers, backbone, sample_model_dir=''):
    """
    Inference sample from selected models.
    Parameters:
    ----------
    url_sample: string,
    Input data sample.
    labId:  string,
    ID of lab.
    ckpt_number: int,
    Number of checkpoint for selected model.
    model_type: string,
    Type of rnn cell model.
    Returns:
    --------
    Label of input data sample.

    """
    # Configure model directory
    if sample_model_dir:
        model_dir = sample_model_dir
    else:
        model_dir = f'./modelDir/{labId}/log_train/{model_type}'

    # Load tokenizer from file
    tokenizer_file = open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb')
    tokenizer = pickle.load(tokenizer_file)
    tokenizer_file.close()

    # Load embeding matrix from file
    embeding_matrix_file = open(os.path.join(
        model_dir, 'embedding_matrix.pkl'), 'rb')
    embeding_matrix = pickle.load(embeding_matrix_file)
    embeding_matrix_file.close()

    # checkpoint path
    if sample_model_dir:
        ckpt_path = os.path.join(model_dir, 'ckpt')
    else:
        ckpt_path = os.path.join(model_dir, 'ckpt-'+str(ckpt_number))

    # Create model and load weights from checkpoint path
    model = get_model(tokenizer=tokenizer, embedding_matrix = embeding_matrix, rnn_cell= backbone, num_neurons= num_neurons, num_layers = num_layers, embed_size=embed_size)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path)

    # Get data sample
    x = get_sample_data(url_sample, tokenizer)

    # Inference sample
    prediction = model(x)
    label_index = tf.argmax(prediction, axis=1)

    # Get label
    label = int2label[np.array(label_index)[0]]

    # Get score of prediction
    score = float(np.array(prediction)[0, label_index][0])
    # yield to backend
    return {
        "label": label,
        "score": score,
        "url": url_sample,
        "model_checkpoint_number": ckpt_number or "Invalid"
    }


if __name__ == '__main__':
    # try:
    infer('http://www.dvdsreleasedates.com/top-movies.php', 'lab2', 1, 'lstm')
    # except:
    # 	pass
