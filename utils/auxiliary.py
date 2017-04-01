# -*- coding: utf-8 -*-
"""Auxiliary functions that support for system."""

from datetime import datetime


def get_fullname(o):
    """get the full name of the class."""
    return '%s.%s' % (o.__module__, o.__class__.__name__)


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)
