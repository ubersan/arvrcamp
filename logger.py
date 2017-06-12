#!/usr/bin/env python3

"""Log strings to files"""

import datetime
import configuration


def _log(filename, string):

    """Logs a string to a file"""

    with open(filename, 'a') as f:

        timestamp = '{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())

        f.write(timestamp + " " + string + "\n")


def log_event(string):

    """Logs string to the event log"""

    if configuration.DEBUG_SERVER_LOG:
        _log(configuration.DEBUG_SERVER_LOGFILE_EVENT, string)


def log_error(string):

    """Logs string to the error log"""

    if configuration.DEBUG_SERVER_LOG:
        _log(configuration.DEBUG_SERVER_LOGFILE_ERROR, string)
