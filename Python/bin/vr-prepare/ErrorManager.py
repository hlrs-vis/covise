# Part of the vr-prepare program for dc

# Copyright (c) 2008 Visenso GmbH


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

# define a global ErrorManager

# define custom Exceptions


_globalErrorManager = None
def globalErrorManager(newHandler=None):
    """Assert instance and access to the error manager."""

    global _globalErrorManager
    if _globalErrorManager == None:
        _globalErrorManager = ErrorManager()
    return _globalErrorManager


class ErrorManager(object):
    """
    An error manager to handle custom error messages
    Currently just a stub
    """

    def __init__(self):
        _infoer.function(str(self.__init__))
        #_infoer.write("")


######################
# custom error codes #
######################

NO_ERROR = 0
WRONG_PATH_ERROR = 1
TIMESTEP_ERROR = 2

############################
# custom exception classes #
############################

class CoviseFileNotFoundError(Exception):
    """ Raised, when a .covise file wasnt found """

    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return self.filename


class ConversionError(Exception):
    """ Raised, when e.g. a string couldnt be converted to int """

    def __init__(self, sourceValue, targetType):
        self.value = sourceValue
        self.targetType = targetType

    def __str__(self):
        return str(value)


class OutOfDomainError(Exception):
    """ Raised, when some value gets out of its domain """

    def __init__(self, value, domain):
        self.value = value
        self.domain = domain

    def __str__(self):
        return str(value)

class TimestepFoundError(Exception):
    """ Raised, when a .covise file has timesteps """

    def __init__(self):
        pass

    def __str__(self):
        return str('')
        
        
