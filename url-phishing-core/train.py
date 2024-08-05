from model import Model


async def train(data_dir, learning_rate, epochs, batch_size, val_size, embed_size, num_neurons, num_layers, backbone, model_type, labId):
    """
    Train model
    Parameters:
    -----------
    data_dir: str,
    Training data directory.
    learning_rate: float,
    Learning rate for training model.
    epochs:	int,
    Number of training epochs.
    batch_size: int,
    Batch size of training data.
    val_size: float,
    embed_size: int,
    Size of validation set over training dataset
    model_type: string,
    Type of rnn cells for building model
    labId:	string,
    ID of lab (use for backend)
    Returns:
    --------
    Trained models saved by .ckpt file
    """

    # Call model from Model class for training
    model = Model(labId, model_type, train_data_dir = data_dir, val_size=val_size, embed_size=embed_size,  num_neurons=num_neurons, num_layers = num_layers, backbone=backbone)
    train_output = model.train(learning_rate, epochs, batch_size)
    for res_per_epoch in train_output:
        yield res_per_epoch
