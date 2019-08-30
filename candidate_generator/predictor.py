import csv
import datetime

import SimpleITK as sitk
import math
import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.utils.training_utils import multi_gpu_model
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.python.client import device_lib

import config as config
from helpers.candidate_gen_data_processor import DataProcessor
from helpers.shared_helpers import SharedHelpers
from models import model_unet_multislice

sh = SharedHelpers()


class Predictor:
    """
    Initializes the Candidate Generator predictor class
    """

    def __init__(self, base_dir):
        """
        Initialize the Predictor class
        Args:
            base_dir: Experiment save path

        """
        # Weight path
        self.base_dir = base_dir
        # Other paths
        self.prediction_images_gt_path = config.CANDIDATE_GENERATOR_CONFIG["prediction_images_gt_path"]
        self.prediction_images_path = config.CANDIDATE_GENERATOR_CONFIG["prediction_images_path"]
        self.prediction_images_csv_path = config.CANDIDATE_GENERATOR_CONFIG["prediction_images_csv_path"]
        self.prediction_mask_save_path = config.CANDIDATE_GENERATOR_CONFIG["prediction_mask_save_path"]
        self.classification_patch_size = config.CANDIDATE_CLASSIFIER_CONFIG["patch_size"]
        self.calculate_uncertainty = config.CANDIDATE_GENERATOR_CONFIG["calculate_uncertainty"]
        self.al_uncertainty_save_path = config.CANDIDATE_GENERATOR_CONFIG["al_uncertainty_save_path"]
        self.save_predicted_mask = config.CANDIDATE_GENERATOR_CONFIG["save_predicted_mask"]
        self.slab_size = config.CANDIDATE_GENERATOR_CONFIG["slab_size"]

        self.candidate_gen_model = self.load_model(os.path.join(base_dir, 'model.json'),
                                                   os.path.join(base_dir, 'weights.hdf5'))
        # init some classes
        self.postprocessor = DataProcessor()

        self.max_patch_cutout_size = (64, 64, 7)

        self.patch_margin = 0

    def load_model(self, model_arch, model_weight):
        """
        Load the model and the weights
        :param model_arch:
        :param model_weight:
        :return:
        """
        local_device_protos = device_lib.list_local_devices()
        list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        num_gpus = len(list_of_gpus)

        json_file = open(model_arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model = multi_gpu_model(model, gpus=num_gpus)
        model.load_weights(model_weight)

        return model

    def load_model_and_save_single_gpu_model(self, model_arch, model_weight):
        """
        Load the model and the weights
        :param model_arch:
        :param model_weight:
        :return:
        """
        local_device_protos = device_lib.list_local_devices()
        list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        num_gpus = len(list_of_gpus)

        # Get the model
        model, gpu_model = model_unet_multislice.get_model(self.slab_size, base_dir=self.base_dir,
                                                           save_model=False)
        model_weights_before = model.get_weights()

        gpu_model.load_weights(model_weight)

        model_weights_after = model.get_weights()

        if not np.array_equal(model_weights_before, model_weights_after):
            print("Weights are NOT similar")
        else:
            print("Weights ARE similar")

        # serialize weights to HDF5
        model.save_weights(os.path.join(self.base_dir, "single_gpu_weights.hdf5"))
        print("Saved model to disk")

        return gpu_model

    def predict(self):
        """
        Method that actually runs the prediction
        :return: None
        """
        # Get the list of image files
        image_extension = "_raw.nii.gz"

        if os.path.isfile(self.prediction_images_csv_path):

            try:
                f = open(self.prediction_images_csv_path, 'r')
            except IOError:
                print("Could not read file:", self.prediction_images_csv_path)
                exit(0)
            with f:
                reader = csv.reader(f)
                image_files = [row[1] for row in reader]
                sh.print("Processing CSV file of size ", len(image_files))
        else:
            image_files = [val for sublist in
                           [[os.path.join(i[0], j) for j in i[2] if j.endswith(image_extension)] for i in
                            os.walk(self.prediction_images_path)] for val in
                           sublist]
            sh.print("Processing local file of size ", len(image_files))

        i = 1
        # Loop thru all the files
        for image_file_name in image_files:
            sh.print("Processing: " + image_file_name)
            # Remove file extensions
            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]
            # Get the image volume
            image = sitk.ReadImage(os.path.join(self.prediction_images_path, image_file_name), sitk.sitkFloat32)

            # Get the candidate mask
            try:
                candidate_mask = self.get_candidate_mask(image, source_name)
            except:
                print("Could not process " + image_file_name)

            # # Get Candidates - Saves the candidates in the generate_candidates method
            # # DISABLED
            # candidate_df = self.generate_candidates(candidate_mask, source_name, image_file_name)
            # sh.print("Total Number of Candidates", str(len(candidate_df)))
            #
            # # Save the candidate_df is output is available
            # if self.prediction_mask_save_path:
            #     output = os.path.join(self.prediction_mask_save_path, source_name + ".csv")
            #     candidate_df.to_csv(output, index=True)
            #     sh.print('Candidates saved as {}'.format(output))

            # Progress report
            sh.print('Finished... {0}/{1} images'.format(i, len(image_files)))
            # Counter
            i += 1

    def get_candidate_mask(self, image, source_name):
        """
        Method to generate the candidate mask.
        This is the method we run after init to fire up the mask generator
        :param image:
        :return:
        """
        # Post-processor takes the image does window correction and returns batch
        image_to_batch = self.postprocessor.postprocess(image)
        # Predict
        prediction = self.candidate_gen_model.predict(image_to_batch, batch_size=32)
        # Get the mask image
        prediction = prediction[..., 1]
        prediction = np.transpose(prediction, [0, 2, 1])
        sh.print('Finished analyzing image batch')
        predicted_mask = sitk.GetImageFromArray(prediction)

        # Calculate Uncertainty
        if self.calculate_uncertainty:
            data = prediction.flatten()
            entropy = self.shannon_entropy_of_probabilities(data)
            lc = self.least_confident(data)
            entropy_energy = self.entropy_energy_of_probabilities(data)
            sh.print("Entropy", entropy, "LC", lc, "Entropy Energy", entropy_energy)
            # Training
            with open(self.al_uncertainty_save_path, 'a', newline='') as uncertainty_file:
                fieldnames = ['file_name', 'entropy', 'lc', 'entropy_energy']
                writer = csv.DictWriter(uncertainty_file, fieldnames=fieldnames)
                writer.writerow(
                    {'file_name': source_name, 'entropy': entropy, 'lc': lc, 'entropy_energy': entropy_energy})

        # If resampling was done, we want to match it with the original image
        predicted_mask = self.postprocessor.postprocess_resample_mask(image, predicted_mask)

        # Copy info
        predicted_mask.CopyInformation(image)

        # Apply threshold to convert to binary mask
        predicted_mask = self.postprocessor.hysteresis_threshold(predicted_mask)

        # Save the image is output is available
        if self.save_predicted_mask:
            output = os.path.join(self.prediction_mask_save_path, source_name + "_mask.nii.gz")
            sitk.WriteImage(predicted_mask, output)
            sh.print('Candidate Mask Image Saved at {}'.format(output))

        return predicted_mask

    def generate_candidates(self, candidate_mask, minimum_label_coverage=30.0):
        """
        This method generates all the candidates
        AKA The segmentation part of the algorithm

        :param image: Takes the original SITK volume
        :return: dataframe of candidates with location - see the candidate_df def below
        """
        # Init Local vars
        patch_count = 0
        candidate_df = pd.DataFrame([], columns=["patch_id", "label_no", "bounding_box", "label_area", "centroid",
                                                 "is_patch_valid", 'probability'])

        # Connected Component
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.FullyConnectedOn()
        candidate_mask = cc_filter.Execute(candidate_mask)

        # Get the mask label stat
        label_stat = sitk.LabelShapeStatisticsImageFilter()
        label_stat.Execute(candidate_mask)

        # Label
        for label_number in label_stat.GetLabels():
            area = label_stat.GetPhysicalSize(label_number)
            centroid = label_stat.GetCentroid(label_number)
            centroid_transformed = candidate_mask.TransformPhysicalPointToIndex(centroid)
            bounding_box = label_stat.GetBoundingBox(label_number)
            width = bounding_box[3]
            height = bounding_box[4]
            depth = bounding_box[5]

            # Fork here if emboli is smaller than 64^3
            if (width < self.max_patch_cutout_size[0]) and (height < self.max_patch_cutout_size[1]) and (
                    depth < self.max_patch_cutout_size[2]):

                # print(max_patch_cutout_size, (width, height, depth))
                # Save the image information in df
                candidate_df = candidate_df.append({'patch_id': patch_count, 'label_no': label_number,
                                                    "bounding_box": bounding_box,
                                                    "label_area": area,
                                                    "centroid": centroid, "is_patch_valid": False, "probability": -1},
                                                   ignore_index=True)
                patch_count += 1
            else:
                # print('generate_candidates', label_number, bounding_box)
                # Loop thru the extent of the emboli
                for x_loop in range(int(math.ceil(width / self.max_patch_cutout_size[0])) + 1):
                    for y_loop in range(int(math.ceil(height / self.max_patch_cutout_size[1])) + 1):
                        for z_loop in range(int(math.ceil(depth / self.max_patch_cutout_size[2])) + 1):
                            x1 = int(bounding_box[0] + (x_loop * self.max_patch_cutout_size[0]))
                            y1 = int(bounding_box[1] + (y_loop * self.max_patch_cutout_size[1]))
                            z1 = int(bounding_box[2] + (z_loop * self.max_patch_cutout_size[2]))
                            x2 = min((x1 + self.max_patch_cutout_size[0]), bounding_box[0] + width)
                            y2 = min((y1 + self.max_patch_cutout_size[1]), bounding_box[1] + height)
                            z2 = min((z1 + self.max_patch_cutout_size[2]), bounding_box[2] + depth)

                            cuboid_image = candidate_mask[x1:x2, y1:y2, z1:z2]

                            # Get the mask label stat
                            cuboid_label_stat = sitk.LabelShapeStatisticsImageFilter()
                            cuboid_label_stat.Execute(cuboid_image)

                            if label_number in cuboid_label_stat.GetLabels():
                                cuboid_label_area = cuboid_label_stat.GetPhysicalSize(label_number)
                                coverage = ((cuboid_label_area / area) * 100)

                                if coverage > minimum_label_coverage:
                                    cuboid_centroid = cuboid_label_stat.GetCentroid(label_number)
                                    cuboid_bounding_box = (
                                        x1, y1, z1, self.max_patch_cutout_size[0], self.max_patch_cutout_size[1],
                                        self.max_patch_cutout_size[2])

                                    candidate_df = candidate_df.append(
                                        {'patch_id': patch_count, 'label_no': label_number,
                                         "bounding_box": cuboid_bounding_box,
                                         "label_area": area, "centroid": cuboid_centroid,
                                         "is_patch_valid": False,
                                         "probability": -1}, ignore_index=True)
                                    patch_count += 1

        return candidate_df

    def calculate_detection_rate(self):
        """
        Method that calculates detection rates of the predicted masks against GT
        :return: None
        """
        detections_df = pd.DataFrame([], columns=["image_id", "total_area", "total_coverage", "coverage_percentile",
                                                  "total_number_of_labels",
                                                  "number_of_labels_detected", 'number_of_labels_missed',
                                                  'detection_rate',
                                                  'f1', 'mean_iou'])

        # Get the list of image files
        image_extension = "_raw.nii.gz"
        mask_extension = "_mask.nii.gz"

        total_label_count = 0
        total_detection_count = 0

        if os.path.isfile(self.prediction_images_csv_path):

            try:
                f = open(self.prediction_images_csv_path, 'r')
            except IOError:
                print("Could not read file:", self.prediction_images_csv_path)
                exit(0)
            with f:
                reader = csv.reader(f)
                image_files = [row[1] for row in reader]
                sh.print("Processing CSV file of size ", len(image_files))
        else:
            image_files = [val for sublist in
                           [[os.path.join(i[0], j) for j in i[2] if j.endswith(image_extension)] for i in
                            os.walk(self.prediction_images_path)] for val in
                           sublist]
            sh.print("Processing local file of size ", image_files)

        i = 1
        # Loop thru all the files
        for image_file_name in image_files:
            sh.print("Processing: " + image_file_name)
            # Remove file extensions
            source_name = os.path.basename(os.path.normpath(image_file_name))[:-len(image_extension)]
            # Get the image volume
            image_gt_path = image_file_name[:-len(image_extension)] + mask_extension
            image_pred_mask_path = os.path.join(self.prediction_mask_save_path, source_name + mask_extension)

            if os.path.isfile(image_pred_mask_path):

                # Read
                img_mask = sitk.ReadImage(image_gt_path)

                pred_mask = sitk.ReadImage(image_pred_mask_path)

                # GT mask Stat
                img_stat = sitk.LabelShapeStatisticsImageFilter()
                img_stat.Execute(img_mask)
                labels = img_stat.GetLabels()

                # Pred mask Stat
                pred_stat = sitk.LabelShapeStatisticsImageFilter()
                pred_stat.Execute(pred_mask)
                pred_labels = pred_stat.GetLabels()

                min_labels = min(labels) if len(labels) > 0 else 0
                max_labels = max(labels) if len(labels) > 0 else 0

                min_pred_labels = min(pred_labels) if len(pred_labels) > 0 else 0
                max_pred_labels = max(pred_labels) if len(pred_labels) > 0 else 0

                # Binary threshold filter
                threshold_filter = sitk.BinaryThresholdImageFilter()

                true_binary_mask = threshold_filter.Execute(img_mask, min_labels, max_labels, 1, 0)
                pred_binary_mask = threshold_filter.Execute(pred_mask, min_pred_labels, max_pred_labels, 1, 0)

                annotated_mask_data, annotated_pred_data = self.get_annotation_prediction_pair(true_binary_mask,
                                                                                               pred_binary_mask)

                detections_df, total_label_number, label_hit_count = self.get_detection_rate(annotated_mask_data,
                                                                                             annotated_pred_data,
                                                                                             image_file_name,
                                                                                             detections_df)

                total_label_count += total_label_number
                total_detection_count += label_hit_count

            else:
                sh.print('Cant Find the mask: ' + image_pred_mask_path)

            # Progress report
            sh.print('Finished... {0}/{1} images'.format(i, len(image_files)))
            i += 1

        # Save the detections
        output = os.path.join(self.prediction_mask_save_path,
                              "detections_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv")
        detections_df.to_csv(output, index=True)
        sh.print('Detections saved as {}'.format(output))
        sh.print('Total Label Count', total_label_count, 'Total Detection Count', total_detection_count)

    def get_annotation_prediction_pair(self, img_mask, pred_mask):
        annotated_mask_data = []
        annotated_pred_data = []
        index = 0
        for slice_number in range(img_mask.GetSize()[2]):

            mask_slice = img_mask[:, :, slice_number]
            pred_slice = pred_mask[:, :, slice_number]
            # Mask Stat
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(mask_slice)

            if stats.GetNumberOfLabels() > 0:
                annotated_mask_data.append(sitk.GetArrayFromImage(mask_slice))
                annotated_pred_data.append(sitk.GetArrayFromImage(pred_slice))

        return np.array(annotated_mask_data), np.array(annotated_pred_data)

    def get_detection_rate(self, annotated_mask_data, annotated_pred_data, image_id, detections_df):
        annotated_mask_image = sitk.GetImageFromArray(annotated_mask_data)
        annotated_pred_image = sitk.GetImageFromArray(annotated_pred_data)
        # Connected Component
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.FullyConnectedOn()
        img_mask = cc_filter.Execute(annotated_mask_image)

        # Stat
        img_stat = sitk.LabelShapeStatisticsImageFilter()
        img_stat.Execute(img_mask)

        # Binary threshold filter
        threshold_filter = sitk.BinaryThresholdImageFilter()

        total_area = 0
        total_coverage = 0
        label_hit_count = 0
        total_label_number = img_stat.GetNumberOfLabels()

        # Label
        for label_number in img_stat.GetLabels():

            total_label_area = 0
            total_label_coverage = 0
            label_hit_flag = False

            for slice_number in range(img_mask.GetSize()[2]):
                current_mask = img_mask[:, :, slice_number]
                binary_mask = threshold_filter.Execute(current_mask, label_number, label_number, 1, 0)
                binary_mask_stats = sitk.StatisticsImageFilter()
                binary_mask_stats.Execute(binary_mask)

                if binary_mask_stats.GetMaximum() > 0.0:
                    image_1 = binary_mask
                    image_1_stats = sitk.LabelShapeStatisticsImageFilter()
                    image_1_stats.Execute(image_1)
                    total_label_area += image_1_stats.GetPhysicalSize(1)

                    # Ref
                    image_2 = annotated_pred_image[:, :, slice_number]
                    min_max_filter = sitk.MinimumMaximumImageFilter()
                    min_max_filter.Execute(image_2)

                    if min_max_filter.GetMaximum() > 0:
                        image_2 = threshold_filter.Execute(image_2, 1, min_max_filter.GetMaximum(), 1, 0)

                        # Add
                        add_filter = sitk.AddImageFilter()
                        added_image = add_filter.Execute(image_1, image_2)
                        # print "Ref Stat: ", min_max_filter.GetMaximum(), " Slice No: ", slice_number
                        added_image_stats = sitk.LabelShapeStatisticsImageFilter()
                        added_image_stats.Execute(added_image)
                        if added_image_stats.GetNumberOfLabels() > 1:
                            total_label_coverage += added_image_stats.GetPhysicalSize(2)
                            label_hit_flag = True

            total_area += total_label_area
            total_coverage += total_label_coverage
            if label_hit_flag == True:
                label_hit_count += 1

        coverage_percentile = ((total_coverage / total_area) * 100)
        number_of_labels_missed = total_label_number - label_hit_count
        detection_rate = (float(label_hit_count) / float(total_label_number)) * 100.0

        # Calculate IOU
        f1, mean_iou = self.compute_mean_iou(annotated_mask_data, annotated_pred_data)
        sh.print('detection_rate', detection_rate)

        # Save the image information in df
        detections_df = detections_df.append({'image_id': image_id, 'total_area': total_area,
                                              "total_coverage": total_coverage,
                                              "coverage_percentile": coverage_percentile,
                                              "total_number_of_labels": total_label_number,
                                              "number_of_labels_detected": label_hit_count,
                                              "number_of_labels_missed": number_of_labels_missed,
                                              "detection_rate": detection_rate, "f1": f1, "mean_iou": mean_iou,
                                              }, ignore_index=True)

        return detections_df, total_label_number, label_hit_count

    def compute_mean_iou(self, y_true, y_pred):
        # ytrue, ypred is a flatten vector
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        current = confusion_matrix(y_true, y_pred, labels=[1, 0])
        f1 = f1_score(y_true, y_pred, labels=[1, 0])

        # compute mean iou
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)

        return f1, np.mean(IoU)

    def shannon_entropy_of_probabilities(self, array):
        """
        Calculates the Shannon Entropy of probabilities
        :return: Entropy
        """
        array = array.astype(np.float)
        array = np.array(list(filter(lambda x: x != 0, array)))
        pa = array
        # pa = a / a.sum()
        return -np.sum(pa * np.log2(pa))

    def entropy_energy_of_probabilities(self, pa):
        """
        Calculates Entropy energy of probabilities
        :return: Entropy energy of pixels
        """
        pa = pa.astype(np.float)
        pa = np.array(list(filter(lambda x: x != 0, pa)))
        one_minus_pa = 1 - pa
        min_nonzero = np.min(one_minus_pa[np.nonzero(one_minus_pa)])
        one_minus_pa[one_minus_pa == 0] = min_nonzero

        # return -np.sum(pa * np.log2(pa)) - np.sum((1 - pa) * np.log2(1 - pa))
        return -np.sum((pa * np.log2(pa)) - ((one_minus_pa) * np.log2(one_minus_pa)))

    def least_confident(self, array):
        """
        Least Confident Sampling from probabilities
        :param array:
        :return: lc score
        """
        score = -np.max(array)
        return score
