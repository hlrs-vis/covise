"""File: printing.py

Find auxiliaries for printing infos during program-execution.


Copyright (c) 2006 Visenso GmbH

"""


class InfoPrintCapable(object):

    """To write info-strings conveniently.

    Set a class and/or a function and/or a
    python-module for inclusion in the print-out by
    setting the object into the respective state.
    I.e. set the members to the right names.  None
    disables the respective output.

    >>> infoer = InfoPrintCapable()
    >>> assert InfoPrintCapable.masterSwitch == True
    >>> infoer.write('foo')
    >>> infoer.doPrint = True
    >>> infoer.write('foo')
    (info) foo
    >>> infoer.startString = ''
    >>> infoer.write('foo')
    foo
    >>> infoer.startString = '(error)'
    >>> infoer.function = 'fun'
    >>> infoer.write('bar')
    (error)(function "fun") bar
    >>> infoer.module = 'module-name'
    >>> infoer.class_ = 'class-name'
    >>> infoer.function = 'function-name'
    >>> infoer.write('bar')
    (error)(module "module-name")(class "class-name")(function "function-name") bar
    >>> InfoPrintCapable.masterSwitch = False
    >>> infoer.write('bar')
    >>> InfoPrintCapable.masterSwitch = True
    >>> infoer.reset()
    >>> infoer.write('bar')
    (info) bar

    """

    # 'masterSwitch == False' means absolute no printing of any
    # InfoPrintCapable-instance.
    masterSwitch = True # True == on, False == off.

    def __init__(self, pymodule=None, pyclass=None, pyfunc=None):
        self.__resetStartString = '(info)'
        self.doPrint = False
        self.module = pymodule
        self.class_ = pyclass
        self.function = pyfunc
        self.startString = self.__resetStartString


    def write(self, aString):
        if not InfoPrintCapable.masterSwitch or not self.doPrint:
            return
        assert isinstance(self.startString, str)
        resultString = self.startString
        if self.module:
            resultString += '(module "%s")' % self.module
        if self.class_:
            resultString += '(class "%s")' % self.class_
        if self.function:
            resultString += '(function "%s")' % self.function
        if resultString != '':
            resultString += ' '
        print(resultString + aString)

    def reset(self):
        self.startString = self.__resetStartString
        self.module = ''
        self.class_ = ''
        self.function = ''


def _test():
    """Doctest start."""
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    # Start doctest with 'python <name of this file>'
    # or 'python <name of this file> -v'.  The latter
    # call gives more output.
    _test()
