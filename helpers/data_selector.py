import csv
import datetime

import config as config
from helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()


class DataSelector:
    """
    Data Selector Class

    Creates csv files for training, validation and testing.
    The idea is to create 3 files: generator_training, generator_validation and
    generator_test from the raw annotation directory; we want to do this randomly
    at first but eventually give more control to the user.

    """

    def __init__(self):
        """
        Initialize the Data Selector Class

        """
        sh.print("Initializing Data Selector")
        self.training_set_file = config.DATA_SELECTOR_CONFIG['training_set_file']
        self.validation_set_file = config.DATA_SELECTOR_CONFIG['validation_set_file']

    def get_generator_dataset(self):
        """
        This method reads the csv files and returns list of training and validation (eventually test) image set
        :return: List of Training and Validation set
        """
        # Training Set

        try:
            f = open(self.training_set_file, 'r')
        except IOError:
            print("Could not read file:", self.training_set_file)
            exit(0)
        with f:
            reader = csv.reader(f)
            training_set = [row for row in reader]

        # Validation Set
        try:
            f = open(self.validation_set_file, 'r')
        except IOError:
            print("Could not read file:", self.validation_set_file)
            exit(0)
        with f:
            reader = csv.reader(f)
            validation_set = [row for row in reader]

        sh.print(
            "Selected " + self.training_set_file + " for training and " + self.validation_set_file + " for validation set")

        return training_set, validation_set


    def al_get_next_set(self):
        sh.print("Getting Next set")

    def get_timestamp(self, file_name):
        """
        This method takes the date and time string and returns a datetime object
        (WORKS ONLY FOR THE DATA SELECTOR FILE FORMAT)
        :param file_name:
        :return:
        """
        time_stamp = file_name.split("_")[-2].split('.')[0] + "_" + file_name.split("_")[-1].split('.')[0]
        time_stamp = datetime.datetime.strptime(time_stamp, '%Y-%m-%d_%H-%M-%S')
        return time_stamp
