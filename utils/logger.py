# -*- coding: utf-8 -*-
"""configure logger."""

import logging
import logging.config


class Logger:
    @staticmethod
    def get_logger(name=None, level=logging.DEBUG):
        """get the logger's configuration file."""
        logging.config.fileConfig('settings/logging.conf')
        return logging.getLogger(name)
