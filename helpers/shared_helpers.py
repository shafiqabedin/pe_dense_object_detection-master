import datetime

import config as config


class SharedHelpers:
    """
    The Shared Helper Class that shares methods across the board.
    Gets parameters from the config
    """

    verbose = config.DEFAULT_CONFIG['verbose']

    def print(self, *args):
        """
        Verbose print method

        :param message: The message to be printed
        :return: None
        """
        if SharedHelpers.verbose:
            print(datetime.datetime.now(), ": ", *args)
