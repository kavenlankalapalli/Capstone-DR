def train_model(model, train_gen, val_gen, epochs=10):
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    return history
