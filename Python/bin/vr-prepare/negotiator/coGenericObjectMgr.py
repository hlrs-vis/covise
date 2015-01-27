

# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import coKeydObject, TYPE_GENERIC_OBJECT, TYPE_GENERIC_OBJECT_MGR, globalKeyHandler, RUN_ALL
from Utils import mergeGivenParams
from coGRMsg import coGRMsg, coGRGenericParamChangedMsg
import covise
from printing import InfoPrintCapable
import Neg2Gui

### global prints
_infoer = InfoPrintCapable()
_infoer.doPrint =  False

PARAM_TYPE_BOOL = 0
PARAM_TYPE_INT = 1
PARAM_TYPE_FLOAT = 2
PARAM_TYPE_STRING = 3
PARAM_TYPE_VEC3 = 4
PARAM_TYPE_MATRIX = 5

NEXT_PRES_STEP_ALLOWED = "NextPresStepAllowed" # This param will not be stored in vr-prepare. If you use it, it has to be set correctly all the time by the plugin.

class coGenericObjectMgr(coKeydObject):
    """ class to handle covise keys for visability """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_GENERIC_OBJECT_MGR, 'GenericObjectMgr')
        globalKeyHandler().globalGenericObjectMgrKey = self.key
        self.params = coGenericObjectMgrParams()

    def recreate(self, negMsgHandler, parentKey, offset):
        coGenericObjectMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        globalKeyHandler().globalGenericObjectMgrKey = self.key

    def addGenericParamFromRenderer(self, objectName, paramName, paramType, defaultValue):
        # create objects if nescessary
        parentKey = self.key
        currentPath = ""
        currentName = ""
        shrinkingPath = objectName
        while (shrinkingPath != ""):
            pos = shrinkingPath.find(".")
            if (pos > -1):
                currentName = shrinkingPath[0:pos]
                shrinkingPath = shrinkingPath[pos+1:]
            else:
                currentName = shrinkingPath
                shrinkingPath = ""
            if (currentPath != ""):
                currentPath = currentPath + "."
            currentPath = currentPath + currentName
            if (currentPath in self.params.name2key):
                obj = globalKeyHandler().getObject(self.params.name2key[currentPath])
            else:
                obj = Neg2Gui.theNegMsgHandler().internalRequestObject( TYPE_GENERIC_OBJECT, parentKey)
                obj.params.name = currentName
                obj.params.path = currentPath
                self.params.name2key[currentPath] = obj.key
                Neg2Gui.theNegMsgHandler().sendParams(obj.key, obj.params )
            parentKey = obj.key
        # add the param
        obj.addGenericParamFromRenderer(paramName, paramType, defaultValue)

    def changeGenericParamFromRenderer(self, objectName, paramName, value):
        if (objectName in self.params.name2key):
            obj = globalKeyHandler().getObject(self.params.name2key[objectName])
            obj.changeGenericParamFromRenderer(paramName, value)

    def deleteKey(self, key):
        self.params.name2key = dict([(a,b) for a,b in self.params.name2key.items() if b != key])

class coGenericObjectMgrParams(object):
    def __init__(self):
        self.name = 'coGenericObjectMgrParams'
        coGenericObjectMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'name2key' : {} # contains Aa as well as Aa.Bb and Aa.Bb.Cc
        }
        mergeGivenParams(self, defaultParams)




class coGenericObject(coKeydObject):
    """ class to handle viewpoints """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_GENERIC_OBJECT, 'coGenericObject')
        self.params = coGenericObjectParams()

    def __toCorrectType(self, value, paramType):
        if (paramType == PARAM_TYPE_BOOL):
            return (value!="0")
        if (paramType == PARAM_TYPE_INT):
            return int(value)
        if (paramType == PARAM_TYPE_FLOAT):
            return float(value)
        if (paramType == PARAM_TYPE_VEC3):
            return [float(x) for x in value.split("/")]
        if (paramType == PARAM_TYPE_MATRIX):
            return [float(x) for x in value.split("/")]
        return str(value)

    def __toString(self, value, paramType):
        if (paramType == PARAM_TYPE_BOOL):
            if value:
                return "1"
            else:
                return "0"
        if (paramType == PARAM_TYPE_INT):
            return str(value)
        if (paramType == PARAM_TYPE_FLOAT):
            return str(value)
        if (paramType == PARAM_TYPE_VEC3):
            return str(value[0]) + "/" + str(value[1]) + "/" + str(value[2])
        if (paramType == PARAM_TYPE_MATRIX):
            return str(value[ 0]) + "/" + str(value[ 1]) + "/" + str(value[ 2]) + "/" + str(value[ 3]) + "/" + \
                   str(value[ 4]) + "/" + str(value[ 5]) + "/" + str(value[ 6]) + "/" + str(value[ 7]) + "/" + \
                   str(value[ 8]) + "/" + str(value[ 9]) + "/" + str(value[10]) + "/" + str(value[11]) + "/" + \
                   str(value[12]) + "/" + str(value[13]) + "/" + str(value[14]) + "/" + str(value[15])
        return str(value)

    def addGenericParamFromRenderer(self, paramName, paramType, defaultValue):
        # add param or update param if nescessary
        self.params.gpTypes[paramName] = paramType
        self.params.gpDefaultValues[paramName] = self.__toCorrectType(defaultValue, paramType)
        if paramName in self.params.gpValues:
            self.sendParamChangeToCover(paramName) # if we already know this param, we send the current value back to Cover
        else:
            self.params.gpValues[paramName] = self.params.gpDefaultValues[paramName] # if the param is new, we use the default value
        # send to GUI
        Neg2Gui.theNegMsgHandler().sendParams(self.key, self.params )
        # set NEXT_PRES_STEP_ALLOWED
        if (paramName == NEXT_PRES_STEP_ALLOWED):
           self.params.nextPresStep = self.__toCorrectType(defaultValue, PARAM_TYPE_BOOL)

    def changeGenericParamFromRenderer(self, paramName, value):
        self.params.gpValues[paramName] = self.__toCorrectType(value, self.params.gpTypes[paramName])
        # send to GUI
        Neg2Gui.theNegMsgHandler().sendParams(self.key, self.params )
        # set NEXT_PRES_STEP_ALLOWED
        if (paramName == NEXT_PRES_STEP_ALLOWED):
           self.params.nextPresStep = self.__toCorrectType(value, PARAM_TYPE_BOOL)

    def setParams(self, params, negMsgHandler = None, sendToCover = True):
        # ignore NEXT_PRES_STEP_ALLOWED
        params.nextPresStep = self.params.nextPresStep
        if (NEXT_PRES_STEP_ALLOWED in self.params.gpValues.keys()):
           params.gpValues[NEXT_PRES_STEP_ALLOWED] = self.params.gpValues[NEXT_PRES_STEP_ALLOWED]
        # determine changed params
        changedParams = []
        for paramName in params.gpValues.keys():
            if paramName not in self.params.gpValues or (self.params.gpValues[paramName] != params.gpValues[paramName]):
                changedParams.append(paramName)
        # set params
        coKeydObject.setParams(self, params)
        # send changes to Cover
        if sendToCover:
            for paramName in changedParams:
                self.sendParamChangeToCover(paramName)

    def sendParamChangeToCover(self, paramName):
        msg = coGRGenericParamChangedMsg(self.params.path, paramName, self.__toString(self.params.gpValues[paramName], self.params.gpTypes[paramName]))
        covise.sendRendMsg(msg.c_str())

    def recreate(self, negMsgHandler, parentKey, offset):
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        coGenericObjectParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class

        # The path is missing in old projects. Since the name was used before, we just copy it.
        if (self.params.path == ""):
            self.params.path = self.params.name

    def delete(self, isInitialized, negMsgHandler=None):
        globalKeyHandler().getObject(globalKeyHandler().globalGenericObjectMgrKey).deleteKey(self.key)
        return coKeydObject.delete(self, isInitialized, negMsgHandler)

class coGenericObjectParams(object):
    def __init__(self):
        self.name      = 'coGenericObjectParams'
        coGenericObjectParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'path' : "",
            'gpTypes' : {},
            'gpDefaultValues' : {}, # values are stored in the correct type
            'gpValues' : {},        # values are stored in the correct type
            'nextPresStep': True
        }
        mergeGivenParams(self, defaultParams)

