import SimpleITK as sitk
import numpy as np
import os
import pandas

import config as config
from helpers.data_selector import DataSelector
from helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()


class DataProcessor:
    """
    Candidate Generator Data Pre Processor
    Creates multi-slice dataset
    use_windowing: Yes or No to windowing
    resample: Resample with SITK takes a long time - use wisely
    validation_split: How much should be validation and train
    slab_depth: Slab depth
    window_minimum: Minimum window value
    window_maximum: Maximum window value

    """

    def __init__(self):
        """
        Initialize the Pre Processor class

        """
        self.slab_size = config.CANDIDATE_GENERATOR_CONFIG['slab_size']
        self.use_windowing = config.CANDIDATE_GENERATOR_CONFIG['use_windowing']
        self.resample = config.CANDIDATE_GENERATOR_CONFIG['resample']
        self.z_spacing = config.CANDIDATE_GENERATOR_CONFIG['z_spacing']
        self.window_minimum = config.CANDIDATE_GENERATOR_CONFIG['window_minimum']
        self.window_maximum = config.CANDIDATE_GENERATOR_CONFIG['window_maximum']
        self.dilation_kernel_radius = config.CANDIDATE_GENERATOR_CONFIG['dilation_kernel_radius']
        self.raw_images_path = config.CANDIDATE_GENERATOR_CONFIG['raw_images_path']
        self.training_images_save_path = config.CANDIDATE_GENERATOR_CONFIG['training_images_save_path']
        self.overwrite_training_images = config.CANDIDATE_GENERATOR_CONFIG['overwrite_training_images']

        self.debug_dir = config.DEFAULT_CONFIG['debug_dir']
        self.threshold = 0.5
        self.hysteresis_lower_threshold = 0.4

        # Init the data selector class
        self.data_selector = DataSelector()

        sh.print("Finished PreProcess Init")

    def postprocess(self, image):
        """
        Data post-process method
        :param image: source
        :return: batch of slabs
        """
        # Resample image
        if self.resample:
            image = self.resample_to_spacing(image, new_spacing=(
                image.GetSpacing()[0], image.GetSpacing()[1], self.z_spacing), interpolation="linear")

        # Windowing
        # Volume Filters
        if self.use_windowing:
            image = sitk.IntensityWindowing(image, self.window_minimum, self.window_maximum)

        # output = os.path.join(self.output_location, 'Candidates_Orig.nii.gz')
        # sitk.WriteImage(image, output)

        # Convert volume to batches of slab
        image_to_batch = self.get_batch_from_image(image)

        return image_to_batch



    def resample_to_spacing(self, image, new_spacing=(1.0, 1.0, 1.0), interpolation="linear"):
        """
        Resamples image to uniform spacing
        :param image: Image to be resampled
        :param interpolation: Interpolation used (linear for image and nearest for mask)
        :return: Resampled image
        """

        if interpolation is "linear":
            interpolator = sitk.sitkLinear
        elif interpolation is "nearest":
            interpolator = sitk.sitkNearestNeighbor
        resampled_image = self.sitk_resample_to_spacing(image, new_spacing, interpolator=interpolator)

        return resampled_image

    def sitk_resample_to_spacing(self, image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear,
                                 default_value=0.):

        zoom_factor = np.divide(image.GetSpacing(), new_spacing)
        new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
        offset = self.calculate_origin_offset(new_spacing, image.GetSpacing())
        reference_image = self.sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                                    origin=image.GetOrigin() + offset, default_value=default_value)
        return self.sitk_resample_to_image(image, reference_image, interpolator=interpolator,
                                           default_value=default_value)

    def calculate_origin_offset(self, new_spacing, old_spacing):
        return np.subtract(new_spacing, old_spacing) / 2

    def sitk_resample_to_image(self, image, reference_image, default_value=0., interpolator=sitk.sitkLinear,
                               transform=None,
                               output_pixel_type=None):
        if transform is None:
            transform = sitk.Transform()
            transform.SetIdentity()
        if output_pixel_type is None:
            output_pixel_type = image.GetPixelID()
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetTransform(transform)
        resample_filter.SetOutputPixelType(output_pixel_type)
        resample_filter.SetDefaultPixelValue(default_value)
        resample_filter.SetReferenceImage(reference_image)
        return resample_filter.Execute(image)

    def sitk_new_blank_image(self, size, spacing, direction, origin, default_value=0.):
        # print("sitk_new_blank_image", size, default_value)
        image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        image.SetOrigin(origin)
        return image

    def remove_files(self, path):
        """
        Remove all the file sin the path
        :param path: Path to look into
        :return: None
        """
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                sh.print(e)

    def hysteresis_threshold(self, image):
        patch = image >= self.threshold
        patch_h = image >= self.hysteresis_lower_threshold

        connected = sitk.ConnectedComponentImageFilter()
        patch_h_labeled = connected.Execute(patch_h)

        label_shape = sitk.LabelShapeStatisticsImageFilter()
        label_shape.Execute(patch_h_labeled)

        old_labels = label_shape.GetLabels()

        masker = sitk.MaskImageFilter()
        patch_h_l_masked = masker.Execute(patch_h_labeled, patch)

        label_shape.Execute(patch_h_l_masked)
        new_labels = label_shape.GetLabels()

        l = 1
        label_map = {}

        for label in old_labels:
            if label in new_labels:
                label_map[label] = l
                l += 1
            else:
                label_map[label] = 0

        label_change = sitk.ChangeLabelImageFilter()
        output = label_change.Execute(patch_h_labeled, label_map)

        return output

    def get_batch_from_image(self, image):
        """
        Method to convert the entire volume into a batch (i.e. 512,512,9,264)
        :param image: Original SITK Image
        :return: Batches of slab
        """
        slab_number = self.slab_size[2]

        image_data = sitk.GetArrayFromImage(image)
        image_data = np.transpose(image_data, [2, 1, 0])

        imagewidth = image.GetSize()[0]
        imageheight = image.GetSize()[1]
        imageslices = image.GetSize()[2]

        data = np.zeros((imageslices, imagewidth, imageheight, slab_number), dtype=np.float32)

        for i in np.arange(imageslices):
            slab = np.zeros((imagewidth, imageheight, slab_number), dtype=np.float32)
            half_slab_num = int(slab_number / 2)
            a = np.arange(i - half_slab_num, i - half_slab_num + slab_number).astype(np.int)
            ind = (np.argwhere((a >= 0) & (a < imageslices))).flatten()
            min_ind = max(i - half_slab_num, 0)
            max_ind = min(i - half_slab_num + slab_number, imageslices)
            slab[..., ind] = image_data[..., min_ind:max_ind].astype(np.float32)
            data[i, ...] = slab
        sh.print('Image converted to {} batches of {} slices.'.format(imageslices, slab_number))
        return data

    def postprocess_resample_mask(self, image, mask):
        """
        Data post-process mask resample method
        :param image: source
        :return: resampled mask
        """
        # Resample image
        if self.resample:
            original_spacing = mask.GetSpacing()
            pixel_spacing = [original_spacing[0], original_spacing[1], image.GetSpacing()[2]]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputOrigin(mask.GetOrigin())
            resampler.SetOutputDirection(mask.GetDirection())
            resampler.SetOutputSpacing(pixel_spacing)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetSize(image.GetSize())
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask = resampler.Execute(mask)

        return mask

    def preprocess(self):
        """
        Data pre-process method
        :return:
        """
        # This seems to be an issue with SITK where multithread doesnt work
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

        # Set padding
        padding = int(self.slab_size[2] / 2)
        # print("Min: " + str(config.CANDIDATE_GENERATOR_CONFIG['window_minimum']) + " Max: " + str(
        #     config.CANDIDATE_GENERATOR_CONFIG['window_maximum']))

        # Patch to file mapping df
        col_names = ['patch_id', 'file_id']
        patch_to_file_df = pandas.DataFrame(columns=col_names)

        # Get the list of image files
        # *** We assume that each image has paired mask ***
        image_extension = "_raw.nii.gz"
        mask_extension = "_mask.nii.gz"

        # Select Genertor Data

        training_set, validation_set = self.data_selector.get_generator_dataset()

        # File Count
        i = 1

        # Starting index for images
        main_index = 0

        # Remove previously created files
        if self.overwrite_training_images:
            sh.print('Doing some cleanup...')
            self.remove_files(os.path.join(self.training_images_save_path, "train-images/"))
            self.remove_files(os.path.join(self.training_images_save_path, "train-masks/"))
            self.remove_files(os.path.join(self.training_images_save_path, "validation-images/"))
            self.remove_files(os.path.join(self.training_images_save_path, "validation-masks/"))
        else:

            current_training_filenames = os.listdir(os.path.join(self.training_images_save_path, "validation-images/"))
            current_training_filenames = sorted(current_training_filenames)
            last_filename = current_training_filenames[-1]
            int_conversion = int(last_filename[:-7])
            main_index = int_conversion + 1

            sh.print('Continuing with current dataset from ' + str(main_index) + '...')

        sh.print('Creating training images...')

        for file_type, image_file_name in training_set + validation_set:

            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]
            sh.print('Loading... {0}/{1} images'.format(i, (len(training_set) + len(validation_set))), " :: ",
                     source_name)
            try:

                # Either assume that the pair in in the same directory OR select from the mask files list -
                # doing the first here
                mask_file_name = image_file_name[:-len(image_extension)] + mask_extension
                # Read Files with SITK
                image_file = sitk.ReadImage(image_file_name, sitk.sitkFloat32)
                mask_file = sitk.ReadImage(mask_file_name, sitk.sitkInt8)

                # Volume Filters
                if self.use_windowing:
                    image_file = sitk.IntensityWindowing(image_file, self.window_minimum, self.window_maximum)
                    mid_index = int(image_file.GetSize()[2] / 2)

                    # Debug Images
                    # if self.debug_dir:
                    #     path = os.path.join(self.debug_dir, "window_check_" + str(i) + ".png")
                    #     sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(image_file[:, :, mid_index]), sitk.sitkUInt8), path)

                    # image_data = sitk.GetArrayFromImage(image_file)
                    # print("Min: " + str(np.percentile(image_data, min_percent)))
                    # print("Max: " + str(np.percentile(image_data, max_percent)))
                    # image_file = sitk.IntensityWindowing(image_file, np.percentile(image_data, min_percent),
                    #                                      np.percentile(image_data, max_percent))

                # Resample to uniform spacing
                if self.resample:
                    sh.print("    Before resampling: " + str(image_file.GetSize()) + " " + str(
                        image_file.GetSpacing()))

                    new_z_spacing = mask_file.GetSpacing()[2]
                    if not new_z_spacing == self.z_spacing:
                        new_z_spacing = self.z_spacing
                        resampled_mask = self.resample_to_spacing(mask_file, new_spacing=(
                            mask_file.GetSpacing()[0], mask_file.GetSpacing()[1], new_z_spacing),
                                                                  interpolation="nearest")
                        resampled_image = self.resample_to_spacing(image_file, new_spacing=(
                            image_file.GetSpacing()[0], image_file.GetSpacing()[1], new_z_spacing),
                                                                   interpolation="linear")
                        sh.print("    After resampling: " + str(resampled_image.GetSize()) + " " + str(
                            resampled_image.GetSpacing()))
                    else:
                        resampled_mask = mask_file
                        resampled_image = image_file
                        sh.print("    No resampling: " + str(resampled_image.GetSize()) + " " + str(
                            resampled_image.GetSpacing()))
                else:

                    resampled_mask = mask_file
                    resampled_image = image_file
                    sh.print("    No resampling: " + str(resampled_image.GetSize()) + " " + str(
                        resampled_image.GetSpacing()))

                # Apply dilation if specified
                if self.dilation_kernel_radius > 0.0:
                    sh.print("    Applying Dilation of size " + str(self.dilation_kernel_radius))
                    dilate_filter = sitk.BinaryDilateImageFilter()
                    dilate_filter.SetKernelRadius(self.dilation_kernel_radius)

                    for slice_number in range(resampled_mask.GetSize()[2]):

                        slice = resampled_mask[:, :, slice_number]
                        # Mask Slice Stat
                        dil_stats = sitk.LabelShapeStatisticsImageFilter()
                        dil_stats.Execute(slice)

                        if dil_stats.GetNumberOfLabels() > 0:
                            slice = dilate_filter.Execute(slice)
                            slice_vol = sitk.JoinSeries(slice)
                            resampled_mask = sitk.Paste(resampled_mask, slice_vol, slice_vol.GetSize(),
                                                        destinationIndex=[0, 0, slice_number])

                total_num_slabs_processed = 0

                # Loop thru slices
                for slice_number in range(resampled_mask.GetSize()[2]):
                    # Get the slice data
                    mask = resampled_mask[:, :, slice_number]

                    # Get the number of scalars
                    stats = sitk.LabelShapeStatisticsImageFilter()
                    stats.SetNumberOfThreads(1)
                    stats.Execute(mask)

                    # Append to the array
                    if stats.GetNumberOfLabels() > 0:

                        labels = stats.GetLabels()

                        # Get the binary mask for that label
                        threshold_filter = sitk.BinaryThresholdImageFilter()
                        binary_mask = threshold_filter.Execute(resampled_mask, min(labels), max(labels), 1, 0)

                        # Get Array From Image
                        binary_mask_data = sitk.GetArrayFromImage(binary_mask)
                        binary_mask_data = np.transpose(binary_mask_data, [2, 1, 0])
                        resampled_image_data = sitk.GetArrayFromImage(resampled_image)
                        resampled_image_data = np.transpose(resampled_image_data, [2, 1, 0])

                        # Crop both Mask and Image
                        mask_slab = np.zeros(self.slab_size, dtype=np.float32)
                        image_slab = np.zeros(self.slab_size, dtype=np.float32)
                        half_slab_num = int(self.slab_size[2] / 2)

                        # Calculate index
                        a = np.arange(slice_number - half_slab_num,
                                      slice_number - half_slab_num + self.slab_size[2]).astype(np.int)
                        index = (np.argwhere((a >= 0) & (a < resampled_mask.GetSize()[2]))).flatten()
                        min_ind = max(slice_number - half_slab_num, 0)
                        max_ind = min(slice_number - half_slab_num + self.slab_size[2], resampled_mask.GetSize()[2])

                        # Extract Slabs

                        mask_slab[..., index] = binary_mask_data[..., min_ind:max_ind].astype(np.float32)
                        image_slab[..., index] = resampled_image_data[..., min_ind:max_ind].astype(np.float32)

                        # Convert back to image
                        mask_slab = np.transpose(mask_slab, [2, 1, 0])
                        mask_slab = sitk.GetImageFromArray(mask_slab)

                        image_slab = np.transpose(image_slab, [2, 1, 0])
                        image_slab = sitk.GetImageFromArray(image_slab)

                        # Save images to the directory
                        image_path = os.path.join(self.training_images_save_path,
                                                  "train-images/" + format(main_index, '06') + ".nii.gz")
                        mask_path = os.path.join(self.training_images_save_path,
                                                 "train-masks/" + format(main_index, '06') + ".nii.gz")

                        if file_type == 'V':
                            # print("Validation", image_file_name)
                            image_path = os.path.join(self.training_images_save_path,
                                                      "validation-images/" + format(main_index, '06') + ".nii.gz")
                            mask_path = os.path.join(self.training_images_save_path,
                                                     "validation-masks/" + format(main_index, '06') + ".nii.gz")
                        # Write patches
                        sitk.WriteImage(mask_slab, mask_path)
                        sitk.WriteImage(image_slab, image_path)

                        # Write to the mapping df
                        patch_to_file_df.loc[len(patch_to_file_df)] = [format(main_index, '05'), image_file_name]

                        # Main Idx
                        main_index += 1

                        # Add to total slabs
                        total_num_slabs_processed += 1

                # Keep Track of number processed
                i += 1
                sh.print("    Total Number Of Slabs Processed: " + str(total_num_slabs_processed))

            except:
                sh.print('*' * 200)
                sh.print('Could NOT Process ' + source_name)
                sh.print('*' * 200)

        # Write csv file
        patch_to_file_df.to_csv(os.path.join(self.training_images_save_path, "segmentation_patch_to_file_mapping.csv"),
                                encoding='utf-8', index=False)
        sh.print('Finished saving all Patches.')
