

# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import TYPE_SCENEGRAPH_ITEM, TYPE_SCENEGRAPH_MGR, globalPresentationMgrKey, globalKeyHandler, RUN_ALL, VIS_VRML, globalProjectKey
from VisItem import VisItem, VisItemParams
from Utils import ParamsDiff, mergeGivenParams, transformationMatrix, CopyParams
from coGRMsg import coGRMsg, coGRObjSetTransparencyMsg, coGRObjSetMoveMsg, coGRObjMaterialObjMsg, coGRObjSetMoveSelectedMsg, coGRObjColorObjMsg, coGRObjTransformSGItemMsg, coGRObjShaderObjMsg
from PartTransform import PartTransformParams
import covise
from printing import InfoPrintCapable
import Neg2Gui
import numpy

### global prints
_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

#ID String
SCENEGRAPH_ID_STRING = 'SCENEGRAPH:'
SCENEGRAPH_APPENDIX_ID_STRING = 'SCGR'

SCENEGRAPH_PARAMS_STRING = 'SCENEGRAPHPARAMS:'

# defines for coloring choice
NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3


class coSceneGraphMgr(VisItem):
    """ class to handle covise keys for visability """
    def __init__(self):
        VisItem.__init__(self, TYPE_SCENEGRAPH_MGR, 'SceneGraphMgr')
        globalKeyHandler().globalSceneGraphMgrKey = self.key
        self.params = coSceneGraphMgrParams()
        self.__inRecreation = False
        self.sceneGraphItems_maximumIndex = -1
        global update_counter
        update_counter = -1

    def recreate(self, negMsgHandler, parentKey, offset):
        self.__inRecreation = True
        self.sceneGraphItems_maximumIndex = -1
        coSceneGraphMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        globalKeyHandler().globalSceneGraphMgrKey = self.key
        global update_counter
        update_counter = -1

        self.sendVisibility()


    def registerCOVISEkey( self, covise_keys):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.registerCOVISEkey)
        _infoer.write("")#for printout
        #_infoer.write("%s" % covise_key)

        """ looking for the prefix SCENEGRAPH_ID_STRING """
        coviseKeyFound = covise_keys.find(SCENEGRAPH_ID_STRING)
        if coviseKeyFound == -1:
            return (False, False)

        # Separating the message into a list of lists [[key, nodeClass, parentkey], ...]
        key_list = covise_keys.split('\t')
        del key_list[0] # idString
        relations = [s.split(";;") for s in key_list if s.count(";;") == 2]

        #######################################################################################
        # UPDATE OLD PROJECTS (part 1/2: delete old objects and remember params)
        update = (globalKeyHandler().getObject(globalProjectKey).params.coprjVersion < 2)
        global update_counter
        if update and (update_counter == -1): # do just once for all SceneGraphMgr registrations
            print("Updating SceneGraphItems")
            update_counter = 0
            # prepare
            global update_oldindex2params
            update_oldindex2params = {}
            global update_oldindex2params_steps
            update_oldindex2params_steps = {}
            # store params
            for coverkey, key in iter(self.params.openCOVER2key.items()):
                oldindex = int(coverkey[coverkey.rfind("_")+1:])
                update_oldindex2params[oldindex] = CopyParams(globalKeyHandler().getObject(key).params)
                for step in globalKeyHandler().getObject(globalPresentationMgrKey).objects:
                    if not step in update_oldindex2params_steps:
                        update_oldindex2params_steps[step] = {}
                    if key in step.params.status:
                        update_oldindex2params_steps[step][oldindex] = CopyParams(step.params.status[key])
                    else:
                        update_oldindex2params_steps[step][oldindex] = None
            # delete objects
            for key in self.params.openCOVER2key.values():
                obj = globalKeyHandler().getObject(key)
                if obj:
                    obj.delete(True, Neg2Gui.theNegMsgHandler())
            # create set of parents
            global update_setOfParents
            update_setOfParents = frozenset([r[2] for r in relations])
        #######################################################################################

        isRoot = True
        for relation in relations:

            __covise_key = relation[0]
            __covise_name = __covise_key[0:__covise_key.find("_"+SCENEGRAPH_APPENDIX_ID_STRING+"_")]
            __covise_index = int(__covise_key[__covise_key.rfind("_")+1:])
            __classname = relation[1]
            __parent_covise_key = relation[2]

            if (__covise_name != "") and (__covise_name + "_" + SCENEGRAPH_APPENDIX_ID_STRING + "_" + str(__covise_index) != __covise_key):
                # we add a * if a node was sent multiple times ("Example_SCGR_07_SCGR_09" -> index is 09, name should be "Example *")
                __covise_name = __covise_name + " *"

            self.sceneGraphItems_maximumIndex = max(self.sceneGraphItems_maximumIndex, __covise_index)

            if not __covise_key in self.params.openCOVER2key:
                # attach to parent if found (to the corresponding VRML_VIS otherwise)
                if __parent_covise_key in self.params.openCOVER2key:
                    parentKey = self.params.openCOVER2key[__parent_covise_key]
                else:
                    parentKey = self.key
                    allVrmlVis = []
                    for item in globalKeyHandler().getAllElements():
                        parentObj = globalKeyHandler().getObject(item)
                        if (parentObj.typeNr == VIS_VRML):
                            allVrmlVis.append(item)
                    if (len(allVrmlVis) > 0):
                        parentKey = allVrmlVis[-1]
                        if update:
                            # in old projects: get the first VRML without children as fallback
                            for i in allVrmlVis:
                                if (len(globalKeyHandler().getObject(i).objects) == 0):
                                    parentKey = i
                                    break
                            # then use the name to identify the VRML
                            for i in allVrmlVis:
                                if __covise_name in globalKeyHandler().getObject(i).params.name:
                                    parentKey = i
                                    break
                        else:
                            # in new projects: simply use the index to identify the VRML
                            for i in allVrmlVis:
                                if (globalKeyHandler().getObject(i).params.sceneGraphItems_startIndex == __covise_index):
                                    parentKey = i
                                    break
                obj = Neg2Gui.theNegMsgHandler().internalRequestObject( TYPE_SCENEGRAPH_ITEM, parentKey, None)

                obj.covise_key = __covise_key
                self.params.openCOVER2key[ __covise_key ] = obj.key

                #######################################################################################
                # UPDATE OLD PROJECTS (part 2/2: use old params)
                if update:
                    # get old index
                    oldindex = -1
                    if (update_counter in update_oldindex2params) and (isRoot or ((__covise_name != "") and (__covise_key in update_setOfParents))):
                        oldindex = update_counter
                        update_counter = update_counter + 1
                    # use existing params
                    if (oldindex != -1):
                        obj.setParams(update_oldindex2params[oldindex])
                        obj.sendAfterRecreate() # send to COVER (since we are using stored params for our new object)
                    # copy coloring options from parent (since we now have the Geode and want the information there)
                    if (__classname == "Geode"):
                        parentObj = globalKeyHandler().getObject(parentKey)
                        if parentObj and isinstance(parentObj, coSceneGraphItem):
                            params = CopyParams(obj.params)
                            params.transparency = parentObj.params.transparency
                            params.transparencyOn = parentObj.params.transparencyOn
                            params.color = parentObj.params.color
                            params.r = parentObj.params.r
                            params.g = parentObj.params.g
                            params.b = parentObj.params.b
                            params.ambient = parentObj.params.ambient
                            params.specular = parentObj.params.specular
                            params.shininess = parentObj.params.shininess
                            obj.setParams(params)
                            obj.sendAfterRecreate() # send to COVER (since we are using stored params for our new object)
                    # add params to presentation steps
                    for step in globalKeyHandler().getObject(globalPresentationMgrKey).objects:
                        if (oldindex == -1):
                            step.params.status[obj.key] = CopyParams(obj.params)
                        elif (update_oldindex2params_steps[step][oldindex] != None):
                            step.params.status[obj.key] = update_oldindex2params_steps[step][oldindex]
                        # copy coloring options from parent (since we now have the Geode and want the information there)
                        if (__classname == "Geode") and (parentKey in step.params.status):
                            parentParams = step.params.status[parentKey]
                            if (isinstance(parentParams, coSceneGraphItemParams)):
                                step.params.status[obj.key].transparency = parentParams.transparency
                                step.params.status[obj.key].transparencyOn = parentParams.transparencyOn
                                step.params.status[obj.key].color = parentParams.color
                                step.params.status[obj.key].r = parentParams.r
                                step.params.status[obj.key].g = parentParams.g
                                step.params.status[obj.key].b = parentParams.b
                                step.params.status[obj.key].ambient = parentParams.ambient
                                step.params.status[obj.key].specular = parentParams.specular
                                step.params.status[obj.key].shininess = parentParams.shininess
                #######################################################################################

                if (__covise_name == ""):
                    obj.params.name = "[unnamed " + __classname + "]"
                else:
                    obj.params.name = __covise_name
                obj.params.nodeClassName = __classname
                obj.sendParams() # send to GUI

            else:
                obj = globalKeyHandler().getObject(self.params.openCOVER2key[__covise_key])
                obj.covise_key = __covise_key

                obj.sendAfterRecreate() # send to COVER
                obj.sendParams() # send to GUI
                
            isRoot = False

        return (True, True)


    #def KeysToDelete(self, msg):
        #_infoer.function = str(self.KeysToDelete)
        #_infoer.write("")
        ##_infoer.write("%s" % msg)

        #keys = []

        #idStringFound = msg.find(SCENEGRAPH_ID_STRING)
        #if idStringFound >-1:
            #""" Separating the message into a list of tuples [key ;; parentkey] """
            #key_list = msg.split('\t')
            #del key_list[0] #idString
            #del key_list[-1] #last tab

            #for item in key_list:
                ##split key from parent key
                #itemlist = item.split(';;')

                #coverKey = itemlist[0]

                #if coverKey in self.params.openCOVER2key:
                    #keys.append( self.params.openCOVER2key[coverKey] )

        #return keys


    def setCOVERParams( self, msg):
        """ setting the COVER params"""
        _infoer.function = str(self.setCOVERParams)
        _infoer.write("")
        #_infoer.write("%s" % msg)

        # if recreated file, we dont want to overwrite the parameters
        if( not self.__inRecreation ):
            idStringFound = msg.find(SCENEGRAPH_PARAMS_STRING)

            if idStringFound >-1:
                """ Separating the message into a list of tuples [key ;; parentkey] """
                key_list = msg.split('\t')
                del key_list[0] #idString
                del key_list[-1] #last tab

                for item in key_list:
                    #split key from parent key
                    itemlist = item.split(';;')

                    key= itemlist[0]
                    transparency = float(itemlist[1])

                    if (key in self.params.openCOVER2key):
                        #set transparency for obj with key
                        obj = globalKeyHandler().getObject(self.params.openCOVER2key[key])
                        if obj:
                            obj.params.transparency = transparency
                            obj.sendParams()


    def moveSelectKey(self, key):
        """ selects a scene graph item and deselects a previous one """

        objNewKey = globalKeyHandler().getObject(key)
        objOldKey = None

        if self.params.moveSelectedSceneGraphItemKey == None:
            msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objNewKey.covise_key, objNewKey.params.isMoveSelected)
            covise.sendRendMsg(msg.c_str())

            self.params.moveSelectedSceneGraphItemKey = key
        elif self.params.moveSelectedSceneGraphItemKey == key:
            msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objNewKey.covise_key, objNewKey.params.isMoveSelected)
            covise.sendRendMsg(msg.c_str())
        else:
            objOldKey = globalKeyHandler().getObject(self.params.moveSelectedSceneGraphItemKey)

            # new key toggled to false, old key currently true -> comes from presentation-mgr, old key was toggled on before
            if objNewKey.params.isMoveSelected == False and objOldKey.params.isMoveSelected == True:
                pass
            # new key toggled to false, old key currently false -> comes from presentation-mgr or user. turn off normally
            elif objNewKey.params.isMoveSelected == False and objOldKey.params.isMoveSelected == False:
                msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objNewKey.covise_key, objNewKey.params.isMoveSelected)
                covise.sendRendMsg(msg.c_str())

                self.params.moveSelectedSceneGraphItemKey = None
            # new key toggled to true, old key currently true -> comes from presentation-mgr or user. turn on normally, deselect old key
            elif objNewKey.params.isMoveSelected == True  and objOldKey.params.isMoveSelected == True:
                # deselect old key
                objOldKey.params.isMoveSelected = False
                objOldKey.sendParams()
                # tell COVER to deselect
                msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objOldKey.covise_key, False)
                covise.sendRendMsg(msg.c_str())

                msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objNewKey.covise_key, objNewKey.params.isMoveSelected)
                covise.sendRendMsg(msg.c_str())

                self.params.moveSelectedSceneGraphItemKey = key
            # new key toggled to true, old key currently false -> comes from presentation-mgr or user. turn on normally
            elif objNewKey.params.isMoveSelected == True  and objOldKey.params.isMoveSelected == False:
                msg = coGRObjSetMoveSelectedMsg(coGRMsg.SET_MOVE_SELECTED, objNewKey.covise_key, objNewKey.params.isMoveSelected)
                covise.sendRendMsg(msg.c_str())

                self.params.moveSelectedSceneGraphItemKey = key

    def deleteKey(self, key):
        """ delete the key from the openCOVER2key dictionary"""
        self.params.openCOVER2key = dict([(ck,k) for ck,k in iter(self.params.openCOVER2key.items()) if k != key])

class coSceneGraphMgrParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name         = 'SceneGraphMgrParams'
        coSceneGraphMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'openCOVER2key' : {},
            'moveSelectedSceneGraphItemKey' : None     # the key of the scene graph item that is selected in COVER for moving
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        if VisItemParams.isStaticParam(self, paramname):
            return True
        return paramname in ["openCOVER2key"]



class coSceneGraphItem(VisItem):
    """ class to handle viewpoints """
    def __init__(self):
        VisItem.__init__(self, TYPE_SCENEGRAPH_ITEM, 'SceneGraph')
        self.params = coSceneGraphItemParams()

    def setParams(self, params, negMsgHandler = None, sendToCover = True):
        realChange = ParamsDiff(self.params, params)
        VisItem.setParams(self, params, negMsgHandler, sendToCover)
        
        needsTransparency = False
        needsShader = False
        
        if 'isMoveable' in realChange:
            self.sendIsMoveable()

        if 'isMoveSelected' in realChange:
            self.sendIsMoveSelected();

        if 'transparency' in realChange or 'transparencyOn' in realChange:
            needsTransparency = True

        if hasattr (params, 'color') and 'color' in realChange and params.color==NO_COLOR:
            self.revertColor()
            needsTransparency = True

        if hasattr (params, 'color') and self.params.color == MATERIAL:
            if 'ambient' in realChange or 'r' in realChange or 'g' in realChange or 'b' in realChange or 'specular' in realChange or 'shininess' in realChange or 'transparency' in realChange or 'transparencyOn' in realChange or 'color' in realChange:
                self.sendMaterial()

        if hasattr (params, 'color') and params.color==RGB_COLOR:
            if 'r' in realChange or 'g' in realChange or 'b' in realChange or 'color' in realChange:
                self.sendColor()

        if ('shaderFilename' in realChange):
            needsTransparency = True
            needsShader = True

        # always send transparency before shader:
        # sendTransparency will ignore any shader transparency but sendShader respects the regular transparency if possible
        if needsTransparency and (params.shaderFilename != ""):
            needsShader = True
        if needsTransparency:
            self.sendTransparency()
        if needsShader:
            self.sendShader()

        # transformation matrix
        if 'rotAngle' in realChange or \
            (self.params.rotAngle != 0 and ('rotX' in realChange or \
                                            'rotY' in realChange or \
                                            'rotZ' in realChange)) or \
            ('transX' in realChange or 'transY' in realChange or 'transZ' in realChange) or \
            ('scaleX' in realChange or 'scaleY' in realChange or 'scaleZ' in realChange):
            self.sendTransform()

    def sendAfterRecreate(self):
        # send to COVER
        self.sendVisibility()
        if not self.params.isMoveable:
            self.sendIsMoveable()
        if self.params.isMoveSelected:
            self.sendIsMoveSelected()
        if self.params.color == MATERIAL:
            self.sendMaterial()
        if self.params.color == RGB_COLOR:
            self.sendColor()
        self.sendTransparency()
        if (self.params.shaderFilename != ""):
            self.sendShader()
        if hasattr(self.params, 'rotAngle') and ((self.params.rotAngle != 0.0) or
                                                 (self.params.transX != 0.0) or (self.params.transY != 0.0) or (self.params.transZ != 0.0) or
                                                 (self.params.scaleX != 1.0) or (self.params.scaleY != 1.0) or (self.params.scaleZ != 1.0)):
            self.sendTransform()

    def sendTransparency(self):
        if (globalKeyHandler().getObject(globalProjectKey).params.originalCoprjVersion >= 3):
            # behavior changed with coprj version 3
            if (len(self.objects) > 0):
                return # we only send transparency messages to leafs (geodes or EOT-nodes)
        if not self.params.transparencyOn:
            transparency = 1.0
        else:
            transparency = self.params.transparency
        # send transparency
        msg = coGRObjSetTransparencyMsg(coGRMsg.SET_TRANSPARENCY, self.covise_key, transparency)
        covise.sendRendMsg(msg.c_str())

    def sendIsMoveable(self):
        msg = coGRObjSetMoveMsg(coGRMsg.SET_MOVE, self.covise_key, self.params.isMoveable)
        covise.sendRendMsg(msg.c_str())

    def sendIsMoveSelected(self):
        globalKeyHandler().getObject(globalKeyHandler().globalSceneGraphMgrKey).moveSelectKey(self.key)

    def sendVisibility(self):
        return VisItem.sendVisibility(self)

    def revertColor(self):
        # set color to grey
        msg = coGRObjColorObjMsg( coGRMsg.COLOR_OBJECT, self.covise_key, -1, -1, -1)
        covise.sendRendMsg(msg.c_str())


    def sendMaterial(self):
        if not self.params.transparencyOn:
            transparency = 1.0
        else:
            transparency = self.params.transparency

        msg = coGRObjMaterialObjMsg( coGRMsg.MATERIAL_OBJECT, self.covise_key, \
                self.params.ambient[0],self.params.ambient[1],self.params.ambient[2], \
                self.params.r, self.params.g, self.params.b, \
                self.params.specular[0],self.params.specular[1],self.params.specular[2], self.params.shininess, transparency)
        covise.sendRendMsg(msg.c_str())

    def sendColor(self):
        msg = coGRObjColorObjMsg( coGRMsg.COLOR_OBJECT, self.covise_key, self.params.r, self.params.g, self.params.b)
        covise.sendRendMsg(msg.c_str())


    def recreate(self, negMsgHandler, parentKey, offset):
        coSceneGraphItemParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        VisItem.recreate(self, negMsgHandler, parentKey, offset)

    # send transform matrix to cover
    def sendTransform(self):
        matrix = transformationMatrix(self.params)
        msg = coGRObjTransformSGItemMsg(self.covise_key, *numpy.reshape(matrix, (16,)))
        covise.sendRendMsg(msg.c_str())

    def sendShader(self):
        if (len(self.objects) > 0):
            return # we only send shader messages to leafs (geodes or EOT-nodes)
        msg = coGRObjShaderObjMsg(coGRMsg.SHADER_OBJECT, self.covise_key, self.params.shaderFilename, "", "", "", "", "", "", "", "", "")
        covise.sendRendMsg(msg.c_str())

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            scgrMgr = globalKeyHandler().getObject(globalKeyHandler().globalSceneGraphMgrKey)
            if scgrMgr: # if no scgrMgr exists, it has already been deleted (no need to delete the key)
                scgrMgr.deleteKey(self.key)
        return VisItem.delete(self, isInitialized, negMsgHandler)

class coSceneGraphItemParams(VisItemParams, PartTransformParams):
    def __init__(self):
        VisItemParams.__init__(self)
        PartTransformParams.__init__(self)
        self.name   = 'SceneGraphItem'
        self.isVisible = True
        coSceneGraphItemParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        PartTransformParams.mergeDefaultParams(self) # call mergeDefaultParams of PartTransformParams since coSceneGraphItem does not inherit from PartTransform
        defaultParams = {
            'isMoveable' : True,
            'isMoveSelected' : False,
            'transparency' : 1.0,
            'transparencyOn' : True,
            'color' : 0,
            'r' : 200,
            'g' : 200,
            'b' : 200,
            'ambient' : [180, 180, 180],
            'specular' : [255, 255, 130],
            'shininess' : 16.0,
            'nodeClassName' : "",
            'shaderFilename' : ""
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        if VisItemParams.isStaticParam(self, paramname):
            return True
        return paramname in ["nodeClassName"]
