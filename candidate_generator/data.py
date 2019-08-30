import numpy as np
import os

import config as config
from helpers import nifti_generator
from helpers.candidate_gen_data_processor import DataProcessor
from helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()


class DataSet:
    """
    Candidate Generator DataSet class
    """

    def __init__(self, is_trainable):
        """
        Initialize the DataSet class

        """
        self.training_images_path = config.CANDIDATE_GENERATOR_CONFIG['training_images_path']
        self.batch_size = config.CANDIDATE_GENERATOR_CONFIG['batch_size']
        self.slab_size = config.CANDIDATE_GENERATOR_CONFIG['slab_size']
        self.train_steps = 0
        self.validation_steps = 0

        # Pre-Process the data if needed
        if config.CANDIDATE_GENERATOR_CONFIG['preprocessing']:
            preprocessor = DataProcessor()
            preprocessor.preprocess()

        # Get the data
        # Load training data / Generators only if model is trainable
        if is_trainable:
            self.generators = self.get_generators()

    def get_generators(self):
        """
        Get Generators. Returns the image generators.
        """

        sh.print('Creating DataGen...')
        # For Multi Slice
        data_gen_training_args = dict(
            rotation_range=20.0,
            shear_range=0.2,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
            back_forth_flip=True,
            augmentation_probability=0.6,
            image_mode='Multislice',
            data_format='channels_last',
        )

        # For Multi Slice
        data_gen_valid_args = dict(
            horizontal_flip=False,
            vertical_flip=False,
            back_forth_flip=False,
            augmentation_probability=0.0,
            image_mode='Multislice',
            data_format='channels_last',
        )

        # Create generators
        train_image_datagen = nifti_generator.ImageDataGenerator(**data_gen_training_args)
        train_mask_datagen = nifti_generator.ImageDataGenerator(**data_gen_training_args)

        validation_image_datagen = nifti_generator.ImageDataGenerator(**data_gen_valid_args)
        validation_mask_datagen = nifti_generator.ImageDataGenerator(**data_gen_valid_args)

        # Provide the same seed and keyword arguments to the fit and flow methods (from Keras doc)
        sh.print("Generating...")
        seed = 1498
        sh.print(self.slab_size)
        train_image_generator = train_image_datagen.flow_from_single_nifti_directory(
            os.path.join(self.training_images_path, "train-images/"),
            batch_size=self.batch_size,
            target_size=self.slab_size,
            # save_prefix="trimg",
            # save_to_dir="/gpfs/fs0/data/DeepLearning/sabedin/experiemnts/pe/debug",
            shuffle=False,
            seed=seed)
        train_mask_generator = train_mask_datagen.flow_from_single_nifti_directory(
            os.path.join(self.training_images_path, "train-masks/"),
            batch_size=self.batch_size,
            target_size=self.slab_size,
            # save_prefix="trmsk",
            # save_to_dir="/gpfs/fs0/data/DeepLearning/sabedin/experiemnts/pe/debug",
            shuffle=False,
            seed=seed)

        validation_image_generator = validation_image_datagen.flow_from_single_nifti_directory(
            os.path.join(self.training_images_path, "validation-images/"),
            batch_size=self.batch_size,
            target_size=self.slab_size,
            shuffle=False,
            seed=seed)
        validation_mask_generator = validation_mask_datagen.flow_from_single_nifti_directory(
            os.path.join(self.training_images_path, "validation-masks/"),
            batch_size=self.batch_size,
            target_size=self.slab_size,
            shuffle=False,
            seed=seed)

        # Combine generators into one which yields image and masks
        # train_generator = self.combine_generator(train_image_generator, train_mask_generator)
        # validation_generator = self.combine_generator(validation_image_generator, validation_mask_generator)
        train_generator = zip(train_image_generator, train_mask_generator)
        validation_generator = zip(validation_image_generator, validation_mask_generator)

        # Sanity Check
        # idx = 0
        # for images, masks in train_generator:
        #
        #     for i in range(self.batch_size):
        #         print("Processing: " + str(idx))
        #         img_input = np.transpose(images[i, :, :, :], (2, 1, 0))
        #         img_input = sitk.GetImageFromArray(img_input)
        #         sitk.WriteImage(img_input, os.path.join(data_root_path, "proof/image-" + str(idx) + ".nii"))
        #
        #         img_output = np.transpose(masks[i, :, :, :], (2, 1, 0))
        #         img_output = sitk.GetImageFromArray(img_output)
        #         sitk.WriteImage(img_output, os.path.join(data_root_path, "proof/mask-" + str(idx) + ".nii"))
        #
        #         idx += 1

        sh.print("Train Sample Size: " + str(train_image_generator.samples) + " - Train Batch Size: " + str(
            train_image_generator.batch_size))
        sh.print("Train Sample Size: " + str(train_mask_generator.samples) + " - Train Batch Size: " + str(
            train_mask_generator.batch_size))
        sh.print(
            "Validation Sample Size: " + str(validation_image_generator.samples) + " - Validation Batch Size: " + str(
                validation_image_generator.batch_size))
        sh.print(
            "Validation Sample Size: " + str(validation_mask_generator.samples) + " - Validation Batch Size: " + str(
                validation_mask_generator.batch_size))

        # Calculate Train and Validation steps
        # print(train_image_generator)
        self.train_steps = (np.ceil(train_image_generator.samples / train_image_generator.batch_size)) + 1
        self.validation_steps = (np.ceil(
            validation_image_generator.samples / validation_image_generator.batch_size)) + 1

        return train_generator, validation_generator

    @staticmethod
    def combine_generator(gen1, gen2):
        """
        Combines the Mask and Image Generators
        """

        while True:
            yield (gen1.next(), gen2.next())
