
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

try:
    import cPickle as pickle
except:
    import pickle

import os.path

from vrpconstants import (
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE,
    )


# Very General Bases
class Named(object):
    def __init__(self, aName=''):
        self.name = aName


class NamedWithFlag(Named):
    def __init__(self, aName='', aFlag=False):
        Named.__init__(self, aName)
        self.flag = aFlag


class NamedList(list):
    def __init__(self, aName=''):
        self.name = aName

    def __str__(self):
        return self.name + ', ' + super(NamedList, self).__str__()

    def __eq__(self, other):
        if other == None:
            return False
        return self.name == other.name and list.__eq__(self, other)

    def __ne__(self, other):
        return not self == other


class NamedWithFlagList(NamedList):
    def __init__(self, aName='', aFlag=False):
        NamedList.__init__(self, aName)
        self.flag = aFlag



# Covise-case-file

class CoviseCaseFileItem:

    """One item keeping geometry and potentially variables.

    ATTENTION: TAKE SPECIAL CARE WITH CHANGES OF THIS
    FILE.  This class _is_ part of the
    covise-case-file-definition.

    """

    def __init__(self, name, dimensionality, geometryFileName=''):
        self.variables_ = []
        assert dimensionality in [GEOMETRY_2D, GEOMETRY_3D]
        self.dimensionality_ = dimensionality
        self.name_ = name
        self.geometryFileName_ = os.path.basename(geometryFileName)


    def addVariableAndFilename(
        self, varName, file='', variableDimension=SCALARVARIABLE):
        assert variableDimension in [SCALARVARIABLE, VECTOR3DVARIABLE]
        self.variables_.append((varName, os.path.basename(file), variableDimension))

    def setDimensionality(self, dimensionality):
        assert dimensionality in [GEOMETRY_2D, GEOMETRY_3D]
        self.dimensionality_ = dimensionality


    def __str__(self):
        dimensionalityToString = {
            GEOMETRY_2D:'GEOMETRY-2D',
            GEOMETRY_3D:'GEOMETRY-3D'}
        variableDimensionToString = {
            SCALARVARIABLE: 'SCALARVARIABLE',
            VECTOR3DVARIABLE: 'VECTOR3DVARIABLE'}
        ret = 'DATAITEM: ' + self.name_ +'\n'
        ret += '-----------------------------------\n'
        ret += '  ' + dimensionalityToString[self.dimensionality_]+'\n'
        ret += '  GEOMETRY ' + self.geometryFileName_ +'\n'
        for ii in self.variables_:
            ret += '  VAR: (is "' + variableDimensionToString[
                ii[2]] + '") ' + ii[0] + ', ' + ii[1] + '\n'
        return ret

    def __eq__(self, other):
        if not other:
            return False
        return ( # Parenthesis is just for formatting.
            self.variables_ == other.variables_ or
            self.dimensionality_ == other.dimensionality_ or
            self.name_ == other.name_ or
            self.geometryFileName_ == other.geometryFileName_)

    def __ne__(self, other):
        return not (self == other)


class CoviseCaseFile(object):

    """1:1 of a covise-case-file.

    ATTENTION: TAKE SPECIAL CARE WITH CHANGES OF THIS
    FILE.  This class _is_ the
    covise-case-file-definition.

    """

    def __init__(self):
        self.version = 1
        self.items_ = []


    def add(self, i):
        self.items_.append(i)


    def __str__(self):
        ret = ''
        for ii in self.items_:
            ret += ii.__str__()
        return ret

    def __eq__(self, other):
        if not other:
            return False
        return self.items_ == other.items_

    def __ne__(self, other):
        return not (self == other)



# Convenience (more or less convenience versions of covise-case-files)

class NameAndCoviseCase(object):

    def __init__(self):
        self.name = ''
        self.case = CoviseCaseFile()
        self.pathToCaseFile = ''

    def setFromFile(self, aFilename):
        """Replace instance-data with case-data from file.
        The name of the instance will be set to the
        filename without path.
        """

        #auxName= os.path.basename(aFilename)
        #self.name = auxName[0: auxName.rfind('.')]
        self.name = os.path.basename(aFilename)

        self.pathToCaseFile = os.path.dirname(aFilename)

        inputFile = open(aFilename, 'rb')
        self.case = pickle.load(inputFile)
        inputFile.close()


class VariableWithFileReference(object):

    def __init__(self, filename, name, variableDimension):
        assert variableDimension in [SCALARVARIABLE, VECTOR3DVARIABLE]
        self.filename = filename
        self.name = name
        self.variableDimension = variableDimension

    def __eq__(self, other):
        if not other:
            return False
        return (
            self.filename == other.filename
            and self.name == other.name
            and self.variableDimension == other.variableDimension)

    def __ne__(self, other):
        return not self == other

class PartWithFileReference(object):

    def __init__(self, filename, name):
        self.filename = filename
        self.name = name
        self.variables = []


    def __eq__(self, other):
        if not other:
            return False
        return (
            self.filename == other.filename
            and self.name == other.name
            and self.variables == other.variables)

    def __ne__(self, other):
        return not self == other


class DimensionSeperatedCase(object):

    __nameOf2dList = 'Geometry (2D Parts)'
    __nameOf3dList = 'Grids (3D Parts)'

    def __init__(self, name=''):
        cls = self.__class__
        self.parts2d = NamedList(cls.__nameOf2dList)
        self.parts3d = NamedList(cls.__nameOf3dList)
        self.name = name


    def getNum2dParts(self):
        return len(self.parts2d)

    def getNum3dParts(self):
        return len(self.parts3d)


    def __eq__(self, other):
        if not other:
            return False
        return (
        self.name == other.name
        and self.parts2d == other.parts2d
        and self.parts3d == other.parts3d)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.name + ', ' + str(self.parts2d) + ', ' + str(self.parts3d)


# Transformer

def extractPartWithFileFromCoviseCaseFileItem(thing, path2CaseFile=''):
    p = PartWithFileReference(
        os.path.join(path2CaseFile, thing.geometryFileName_), thing.name_)
    for variable in thing.variables_:
        p.variables.append(
            VariableWithFileReference(
            os.path.join(path2CaseFile, variable[1]) , variable[0], variable[2]))
    return p


def coviseCase2DimensionSeperatedCase(aCoviseCase, name=None, path2CaseFile=''):
    dsc = DimensionSeperatedCase()
    if not name == None:
        dsc.name = name
    for e in aCoviseCase.items_:
        if GEOMETRY_2D == e.dimensionality_:
            dsc.parts2d.append(
                extractPartWithFileFromCoviseCaseFileItem(e, path2CaseFile))
        elif GEOMETRY_3D == e.dimensionality_:
            dsc.parts3d.append(
                extractPartWithFileFromCoviseCaseFileItem(e, path2CaseFile))
        else:
            assert not 'Reached unreachable point!'
    return dsc


def extractNameAndCheckIndicatoredList(aDSC):

    """Semantic is in here that distinguishes the checkable items.

    Variables below in aDSC are ignored.

    """

    theList = NamedWithFlagList(aDSC.name)
    subList = NamedWithFlagList(aDSC.parts2d.name)
    for e in aDSC.parts2d:
        subList.append(NamedWithFlagList(e.name, True))
    theList.append(subList)
    subList = NamedWithFlagList(aDSC.parts3d.name)
    for e in aDSC.parts3d:
        subList.append(NamedWithFlagList(e.name, True))
    theList.append(subList)
    return theList

def getVectorVariableNames(aPart):
    return [variable.name for variable in aPart.variables
            if variable.variableDimension == VECTOR3DVARIABLE]

def getScalarVariableNames(aPart):
    return [variable.name for variable in aPart.variables
            if variable.variableDimension == SCALARVARIABLE]

# eof
