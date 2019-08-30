"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import print_function

import threading

import SimpleITK as sitk
import numpy as np
import os
from keras import backend as K
from six.moves import range


def transform_image(image, image_mode, rot_interval, translate_interval, zoom_interval,
                    fill_mode=sitk.sitkNearestNeighbor, defaultvalue=-1024):
    if len(zoom_interval) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_interval)

    if image_mode in {'3D', 'Multislice'}:
        transform = sitk.ScaleSkewVersor3DTransform()
        image_center_ind = [image.GetSize()[i] / 2. for i in np.arange(0, image.GetDimension())]
        image_center = image.TransformContinuousIndexToPhysicalPoint(tuple(image_center_ind))
        # transform.SetCenter(image_center)
        if rot_interval:
            theta = np.pi / 180 * np.random.uniform(-rot_interval, rot_interval)
        else:
            theta = 0

        if translate_interval:
            tx = np.random.uniform(-translate_interval, translate_interval) * \
                 image.GetSize()[0] * image.GetSpacing()[0]
            ty = np.random.uniform(-translate_interval, translate_interval) * \
                 image.GetSize()[1] * image.GetSpacing()[1]
        else:
            tx = 0
            ty = 0

        zoom = np.random.uniform(zoom_interval[0], zoom_interval[1])

        if image_mode == '3D':
            tz = np.random.uniform(-translate_interval, translate_interval) * \
                 image.GetSize()[2] * image.GetSpacing()[2]
            vector = generate_random_unit_vector()
            scale = np.array([zoom, zoom, zoom])

        if image_mode == 'Multislice':
            tz = 0
            vector = np.array([0, 0, 1], dtype=np.dtype('d'))
            scale = np.array([zoom, zoom, 1], dtype=np.dtype('d'))

        translation = np.array([tx, ty, tz], dtype=np.dtype('d'))
        transform.SetTranslation(translation)
        transform.SetRotation(vector, theta)
        transform.SetScale(scale)

    if image_mode == '2D':
        transform = sitk.Similarity2DTransform()
        image_center_ind = [image.GetSize()[i] / 2. for i in np.arange(0, image.GetDimension())]
        image_center = image.TransformContinuousIndexToPhysicalPoint(tuple(image_center_ind))
        if rot_interval:
            theta = np.pi / 180 * np.random.uniform(-rot_interval, rot_interval)
        else:
            theta = 0
        transform.SetAngle(theta)

        if translate_interval:
            tx = np.random.uniform(-translate_interval, translate_interval) * \
                 image.GetSize()[0] * image.GetSpacing()[0]
            ty = np.random.uniform(-translate_interval, translate_interval) * \
                 image.GetSize()[1] * image.GetSpacing()[1]
        else:
            tx = 0
            ty = 0

        translation = np.array([tx, ty])
        transform.SetTranslation(translation)

        zoom = np.random.uniform(zoom_interval[0], zoom_interval[1])
        scale = np.array([zoom, zoom])
        transform.SetScale(scale)
    transform.SetCenter(image_center)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(fill_mode)
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(transform)
    resampler.SetSize(image.GetSize())
    #    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(defaultvalue)
    outimage = resampler.Execute(image)

    return outimage


def generate_random_unit_vector():
    z = np.random.uniform(-1, 1)
    theta = np.random.uniform(0, 2 * np.pi)
    ct = np.cos(theta)
    st = np.sin(theta)
    z2s = np.sqrt(1 - z * z)
    return np.array([z2s * ct, z2s * st, z])


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def img_to_array(img, image_mode, data_format=None):
    """Converts a SImpleITK Image instance to a Numpy array.

    # Arguments
        img: SimpleITK Image instance.
        data_format: Image data format.

    # Returns
        A 3D or 4D Numpy array depending on the image dimensions.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, depth, channel)
    # or (channel, height, width, depth)
    # but original SimpleITK image has format (depth, width, height)
    x = sitk.GetArrayFromImage(img)
    if len(x.shape) == 3:
        if image_mode == '3D':
            x = np.transpose(x, (2, 1, 0))
            if data_format == 'channels_first':
                x = np.expand_dims(x, 0)
            else:
                x = np.expand_dims(x, 3)
        if image_mode == 'Multislice':
            if data_format == 'channels_first':
                x = np.transpose(x, (0, 2, 1))
            else:
                x = np.transpose(x, (2, 1, 0))
    elif len(x.shape) == 2:
        if image_mode == '2D':
            x = np.transpose(x, (1, 0))
        if data_format == 'channels_first':
            x = np.expand_dims(x, 0)
        else:
            x = np.expand_dims(x, 2)
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def array_to_image(array, image_mode, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    x = array.copy()
    if len(x.shape) == 4:
        if image_mode == '3D':
            if data_format == 'channels_first':
                x = x[0, ...]
            else:
                x = x[..., 0]
            x = np.transpose(x, (2, 1, 0))
    elif len(x.shape) == 3:
        if image_mode == 'Multislice':
            if data_format == 'channels_first':
                x = np.transpose(x, (0, 2, 1))
            else:
                x = np.transpose(x, (2, 1, 0))
        if image_mode == '2D':
            if data_format == 'channels_first':
                x = x[0, ...]
            else:
                x = x[..., 0]
            x = np.transpose(x, (1, 0))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    img = sitk.GetImageFromArray(x, False)
    return img


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 image_mode='3D',
                 rotation_range=0.,
                 shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 fill_mode=sitk.sitkNearestNeighbor,
                 defaultval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 back_forth_flip=False,
                 preprocessing_function_SimpleITK=None,
                 postprocessing_function_numpy=None,
                 data_format=None,
                 augmentation_probability=0.,
                 ):
        if data_format is None:
            data_format = K.image_data_format()
        if image_mode not in {'2D', '3D', 'Multislice'}:
            raise ValueError('Invalid image mode:', image_mode,
                             '; expected "2D" or "3D" or "Multislice".')
        self.image_mode = image_mode
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.defaultval = defaultval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.back_forth_flip = back_forth_flip
        self.preprocessing_function_SimpleITK = preprocessing_function_SimpleITK
        self.postprocessing_function_numpy = postprocessing_function_numpy
        self.augmentation_probability = augmentation_probability

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow_from_nifti_directory(self, directory,
                                  target_size=(64, 64, 64), extension='nii.gz',
                                  classes=None, class_mode='categorical',
                                  batch_size=32, shuffle=True, seed=None,
                                  save_to_dir=None,
                                  save_prefix='',
                                  follow_links=False):
        return NiftiDirectoryIterator(
            directory, self,
            target_size=target_size, extension=extension,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            follow_links=follow_links)

    def flow_from_single_nifti_directory(self, directory,
                                         target_size=(64, 64, 64), extension='nii.gz',
                                         batch_size=32, shuffle=True, seed=None,
                                         save_to_dir=None,
                                         save_prefix='',
                                         follow_links=False):
        return NiftiSingleDirectoryIterator(
            directory, self,
            target_size=target_size, extension=extension,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            follow_links=follow_links)

    def random_transform(self, x):
        """Randomly augment a single image tensor.

        # Arguments
            x: 2D, 3D single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single 2D, 3D or multislice image in SimpleITK format, so it doesn't have image number at index 0
        # output is a numpy tensor
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if np.random.binomial(1, self.augmentation_probability, 1):

            x = transform_image(x, self.image_mode, self.rotation_range, self.shift_range, self.zoom_range,
                                self.fill_mode, self.defaultval)

            x = img_to_array(x, self.image_mode)

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    x = flip_axis(x, img_col_axis)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    x = flip_axis(x, img_row_axis)

            if self.back_forth_flip:
                if np.random.random() < 0.5:
                    if self.image_mode == '3D':
                        x = flip_axis(x, img_col_axis + 1)
                    if self.image_mode == 'Multislice':
                        x = flip_axis(x, img_channel_axis)
        else:
            x = img_to_array(x, self.image_mode)
        return x


class Iterator(object):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NiftiDirectoryIterator(Iterator):
    """Iterator capable of reading nifti images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 target_size, extension,
                 classes, class_mode,
                 batch_size, shuffle, seed,
                 data_format=None,
                 save_to_dir=None, save_prefix='',
                 follow_links=False):

        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if extension not in {'nii', 'nii.gz'}:
            raise ValueError('Invalid Extension:', extension,
                             '; expected "nii" or "nii.gz".')
        self.extension = extension
        self.image_mode = self.image_data_generator.image_mode
        self.data_format = data_format

        if self.data_format == 'channels_first':
            if self.image_mode == '2D':
                self.image_shape = (1, self.target_size[0], self.target_size[1])
            if self.image_mode == '3D':
                self.image_shape = (1, self.target_size[0], self.target_size[1], self.target_size[2])
            if self.image_mode == 'Multislice':
                self.image_shape = (self.target_size[2], self.target_size[0], self.target_size[1])
        else:
            if self.image_mode == '2D':
                self.image_shape = (self.target_size[0], self.target_size[1], 1)
            if self.image_mode == '3D':
                self.image_shape = (self.target_size[0], self.target_size[1], self.target_size[2], 1)
            if self.image_mode == 'Multislice':
                self.image_shape = (self.target_size[0], self.target_size[1], self.target_size[2])

        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        white_list_formats = {'nii', 'nii.gz'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.samples += 1
        # print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(NiftiDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = sitk.ReadImage(os.path.join(self.directory, fname))
            if img.GetSize()[0] != self.target_size[0] or img.GetSize()[1] != self.target_size[1] or img.GetSize()[2] != \
                    self.target_size[2]:
                raise ValueError('Loaded image size does not match the requested size',
                                 'Received arg: ', self.target_size, 'Image Size', img.GetSize(), 'Filename',
                                 os.path.join(self.directory, fname))
            if self.image_data_generator.preprocessing_function_SimpleITK:
                img = self.image_data_generator.preprocessing_function_SimpleITK(img)
            x = self.image_data_generator.random_transform(img)
            if self.image_data_generator.postprocessing_function_numpy:
                x = self.image_data_generator.postprocessing_function_numpy(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_image(batch_x[i], self.image_mode, self.data_format)
                fname = '{prefix}_{index}_{hash}.nii'.format(prefix=self.save_prefix,
                                                             index=current_index + i,
                                                             hash=np.random.randint(1e4))
                sitk.WriteImage(img, os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class NiftiSingleDirectoryIterator(Iterator):
    """Iterator capable of reading nifti images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 target_size, extension,
                 batch_size, shuffle, seed,
                 data_format=None,
                 save_to_dir=None, save_prefix='',
                 follow_links=False):

        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if extension not in {'nii', 'nii.gz'}:
            raise ValueError('Invalid Extension:', extension,
                             '; expected "nii" or "nii.gz".')
        self.extension = extension
        self.image_mode = self.image_data_generator.image_mode
        self.data_format = data_format

        if self.data_format == 'channels_first':
            if self.image_mode == '2D':
                self.image_shape = (1, self.target_size[0], self.target_size[1])
            if self.image_mode == '3D':
                self.image_shape = (1, self.target_size[0], self.target_size[1], self.target_size[2])
            if self.image_mode == 'Multislice':
                self.image_shape = (self.target_size[2], self.target_size[0], self.target_size[1])
        else:
            if self.image_mode == '2D':
                self.image_shape = (self.target_size[0], self.target_size[1], 1)
            if self.image_mode == '3D':
                self.image_shape = (self.target_size[0], self.target_size[1], self.target_size[2], 1)
            if self.image_mode == 'Multislice':
                self.image_shape = (self.target_size[0], self.target_size[1], self.target_size[2])
        # print(self.image_shape)
        white_list_formats = {'nii', 'nii.gz'}

        # first, count the number of samples and classes
        self.samples = 0

        # if not classes:
        #    classes = []
        #
        #   for subdir in sorted(os.listdir(directory)):
        #        if os.path.isdir(os.path.join(directory, subdir)):
        #            classes.append(subdir)
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        # for root, _, files in _recursive_list(directory):
        #    for fname in files:
        #        is_valid = False
        #        for extension in white_list_formats:
        #            if fname.lower().endswith('.' + extension):
        #                is_valid = True
        #                break
        #        if is_valid:
        #            self.samples += 1
        # print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        i = 0
        # for subdir in classes:
        #    subpath = os.path.join(directory, subdir)
        for root, _, files in _recursive_list(directory):
            for fname in files:
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    i += 1
                    # add filename relative to directory
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path, directory))
        self.samples = i
        super(NiftiSingleDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def next(self):

        """For python 2.x.

        # Returns
            The next batch.
        """

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = sitk.ReadImage(os.path.join(self.directory, fname))
            if img.GetSize()[0] != self.target_size[0] or img.GetSize()[1] != self.target_size[1] or img.GetSize()[2] != \
                    self.target_size[2]:
                raise ValueError('Loaded image size does not match the requested size',
                                 'Received arg: ', self.target_size, 'Image Size', img.GetSize())
            if self.image_data_generator.preprocessing_function_SimpleITK:
                img = self.image_data_generator.preprocessing_function_SimpleITK(img)
            x = self.image_data_generator.random_transform(img)
            if self.image_data_generator.postprocessing_function_numpy:
                x = self.image_data_generator.postprocessing_function_numpy(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_image(batch_x[i], self.image_mode, self.data_format)
                fname = '{prefix}_{index}_{hash}.nii'.format(prefix=self.save_prefix,
                                                             index=current_index + i,
                                                             hash=np.random.randint(1e4))
                sitk.WriteImage(img, os.path.join(self.save_to_dir, fname))
        return batch_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data generator for nifti images.')
    parser.add_argument('-i', dest='input', help='input directory')
    parser.add_argument('-o', dest='output', help='output directory')

    args = parser.parse_args()
    input_directory = args.input
    output_directory = args.output

    aug_args = dict(
        image_mode='3D',
        rotation_range=20.,
        shift_range=0.2,
        shear_range=0.,
        zoom_range=[0.8, 1.2],
        fill_mode=sitk.sitkLinear,
        defaultval=-1024,
        horizontal_flip=True,
        vertical_flip=True,
        back_forth_flip=True,
        augmentation_probability=0.6,
        data_format='channels_last'
    )
    generator = ImageDataGenerator(**aug_args)
    output = generator.flow_from_single_nifti_directory(input_directory, target_size=(64, 64, 64),
                                                        batch_size=32, shuffle=True, seed=None,
                                                        save_to_dir=output_directory,
                                                        save_prefix='')

    batch = 0
    # for image, label in izip(image_list, label_list):
    #    print(image.shape)
    #    print(label.shape)
    #    break
