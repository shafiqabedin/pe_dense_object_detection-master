import sys

import math
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import config as config
from candidate_generator.data import DataSet
from helpers.shared_helpers import SharedHelpers
from models import model_unet_multislice

sh = SharedHelpers()


class Trainer():
    """
    Trainer Class
    """

    def __init__(self, base_dir, is_trainable):
        """
        Initialize the Trainer class
        """
        # Set the base experiment dir
        self.base_dir = base_dir
        self.is_trainable = is_trainable
        self.data = None
        self.workers = config.CANDIDATE_GENERATOR_CONFIG["workers"]
        self.max_queue_size = config.CANDIDATE_GENERATOR_CONFIG["max_queue_size"]
        self.nb_epoch = config.CANDIDATE_GENERATOR_CONFIG["nb_epoch"]
        self.slab_size = config.CANDIDATE_GENERATOR_CONFIG['slab_size']

        # Get Checkpoint
        self.checkpointer = self.get_checkpoint()

        # Get Checkpoint
        self.earlystopper = self.get_earlystopper()

        # Tensorboard
        self.tensorboard = TensorBoard(log_dir=self.base_dir)

        # Custom save Model
        self.best_val_acc = 0
        self.best_val_loss = sys.float_info.max

    def train(self):
        """
        Train function decides which model to fire up given the name of the model.
        """

        # Choose Model

        if config.CANDIDATE_GENERATOR_CONFIG["model_name"] == "UNETMULTISLICE":
            # Get Data
            self.data = DataSet(self.is_trainable)
            # Model
            self.unet_multislice()

    def step_decay(epoch, initial_lrate, drop, epochs_drop):
        """
        # learning rate schedule
        :param initial_lrate:
        :param drop:
        :param epochs_drop:
        :return:
        """
        return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))

    def get_checkpoint(self):
        """
        Returns the checkpoint.
        Returns:
            checkpoint: Keras ModelCheckpoint
        """
        return ModelCheckpoint(
            # filepath=os.path.join(self.base_dir, "{epoch:03d}-{val_loss:.2f}.hdf5"),
            filepath=os.path.join(self.base_dir, "weights.hdf5"),
            verbose=1,
            # monitor='val_loss',  # val_acc
            # mode='min',
            save_best_only=True,  # True
            save_weights_only=True
        )

    def get_earlystopper(self):
        """
        Returns the EarlyStopping.
        Returns:
            earlystopping: Keras EarlyStopping
        """
        return EarlyStopping(patience=20)

    def get_reduce_lr_on_plateau(self):
        """
        Returns the ReduceLROnPlateau.
        Returns:
            reduce_lr_on_plateau: Keras ReduceLROnPlateau
        """
        return ReduceLROnPlateau(factor=0.1, patience=5, verbose=1)

    def train_model(self, model, callbacks=[]):
        """
        Method where the actual training kicks off.
        Args:
            model: The compiled model
            callbacks: List of callbacks
        """

        train_generator, validation_generator = self.data.generators

        # Fit
        model.fit_generator(
            train_generator,
            steps_per_epoch=self.data.train_steps,
            validation_data=validation_generator,
            validation_steps=self.data.validation_steps,
            workers=self.workers,
            use_multiprocessing=False,
            max_queue_size=self.max_queue_size,
            epochs=self.nb_epoch,
            callbacks=callbacks
        )

        return model

    def unet_multislice(self):
        """
        Runs the actual training and prediction finctions
        """
        # Get the model
        model, gpu_model = model_unet_multislice.get_model(self.slab_size, base_dir=self.base_dir,
                                                           save_model=True)

        if self.al_gen_pre_trained_weights:
            sh.print("Loading pre-trained weights from ", self.al_gen_pre_trained_weights)
            gpu_model.load_weights(self.al_gen_pre_trained_weights)

        # Start training
        if self.is_trainable:
            sh.print("Training... ")
            _ = self.train_model(gpu_model, [self.checkpointer, self.tensorboard, self.get_earlystopper()])
