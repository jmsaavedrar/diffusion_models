
import tensorflow as tf
import difmod
import config
import data
import os

if __name__ == '__main__' :
    # load dataset
    train_dataset = data.prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
    val_dataset = data.prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

    model = difmod.DiffusionModel(config.image_size, config.widths, config.block_depth)
    # below tensorflow 2.9:
    # pip install tensorflow_addons
    # import tensorflow_addons as tfa
    # optimizer=tfa.optimizers.AdamW
    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=config.learning_rate, weight_decay=config.weight_decay
        ),
        loss=tf.keras.losses.mean_absolute_error,
    )
    # pixelwise mean absolute error is used as loss

    # save the best model based on the validation KID metric
    checkpoint_path = "checkpoints/diffusion_model"
    if not os.path.exists(os.path.dirname(checkpoint_path)) :
        os.makedirs(os.path.dirname(checkpoint_path))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_kid",
        mode="min",
        save_best_only=True,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=config.num_epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            checkpoint_callback,
        ],
    )