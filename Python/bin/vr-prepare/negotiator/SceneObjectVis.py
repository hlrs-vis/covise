
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import VRPCoviseNetAccess

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet,
    saveExecute)

from VisItem import VisItem, VisItemParams
from coPyModules import PerformerScene
from KeydObject import RUN_ALL, VIS_SCENE_OBJECT, globalKeyHandler
from Utils import ParamsDiff, mergeGivenParams, CopyParams
from coGRMsg import coGRMsg, coGRObjDelMsg, coGRObjMovedMsg, coGRObjGeometryMsg, coGRObjAddChildMsg, coGRObjSetVariantMsg, coGRObjSetAppearanceMsg, coGRObjKinematicsStateMsg
import covise
from Utils import getExistingFilename

import Neg2Gui

from ErrorManager import CoviseFileNotFoundError

import xml.dom.minidom 


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

class SceneObjectVis(VisItem):

    def __init__(self):
        VisItem.__init__(self, VIS_SCENE_OBJECT, self.__class__.__name__)
        self.params = SceneObjectVisParams()
        self.params.isVisible = True

        self.__initBase()

    def __initBase(self):
        self.__connected = False
        self.parentCoviseKeyToSend = 'No key'
        self.performerScene = None
        self.__loaded = False
        
    def __update(self):
        """ __update is called from the run method to update the module parameter before execution
            + update module parameters """
        _infoer.function = str(self.__update)
        _infoer.write(" ")
        if self.performerScene==None:
            self.performerScene = PerformerScene()
            theNet().add(self.performerScene)
        # update params
        self.performerScene.set_modelPath( covise.getCoConfigEntry("vr-prepare.Coxml.ResourceDirectory") + "/coxml/" + self.params.filename )
        self.performerScene.set_scale(self.params.scale)
        if (self.params.backface == True):
            self.performerScene.set_backface('TRUE')
        else:
            self.performerScene.set_backface('FALSE')
        if (self.params.orientation_iv == True):
            self.performerScene.set_orientation_iv('TRUE')
        else:
            self.performerScene.set_orientation_iv('FALSE')
        if (self.params.convert_xforms_iv == True):
            self.performerScene.set_convert_xforms_iv('TRUE')
        else:
            self.performerScene.set_convert_xforms_iv('FALSE')
        _infoer.write(" finished")
        
    def registerCOVISEkey( self, covise_key):
        """ check if object name was created by this visItem
            and if yes store it """
        (registered, firstTime) = VisItem.registerCOVISEkey(self, covise_key)
        if registered: 
            self.sendParent()
            self.sendGeometry()
            self.sendObjMoved()
            self.sendVariants()
            self.sendAppearance()
            self.sendKinematics()
            self.sendChildren()
        return (registered, firstTime)      

    def createdKey(self, key):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.createdKey)
        _infoer.write(" start")
        importKey = self.performerScene.getCoObjName('model')
        posCover  = key.find("_OUT")
        posCover2 = key.find("(")
        if posCover2<posCover: posCover=posCover2
        posImport = importKey.find("_OUT")
        return ( importKey[0:posImport]==key[0:posCover] )

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        if self.performerScene:
            return ConnectionPoint(self.performerScene, 'model')

    def recreate(self, negMsgHandler, parentKey, offset):
        SceneObjectVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        #add offset to children
        newchildren = []
        for child in self.params.children:
            newchildren.append(child+offset)
        self.params.children = newchildren
        if self.params.filename == None:
            raise CoviseFileNotFoundError(self.params.filename)


    def run(self, runmode, negMsgHandler=None):
        _infoer.function = str(self.run)
        _infoer.write(" ")
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")
            self.__update()
            if not self.__connected and self.params.isVisible:
                VisItem.connectToCover( self, self )
                self.__connected=True
            if not self.__loaded:
                saveExecute(self.performerScene)
                self.__loaded=True

    def delete(self, isInitialized, negMsgHandler=None):
        ''' delete this CoviseVis: remove the module '''
        _infoer.function = str(self.delete)
        _infoer.write(" ")

        if isInitialized:

            # Manually remove the object from any parent object it is mounted to.
            # This is not very elegant because when an object (child) is deleted, the unmounting
            # done in OpenCOVER will send a removeChildMessage to the parent. However, because
            # it's a separate process, the child will already be fully deleted in vr-prepare
            # when receiving this message and the coverKey of the child is not known anymore.   
            sceneObjects = [obj for obj in globalKeyHandler().getAllElements().itervalues() if hasattr(obj,"typeNr") and (obj.typeNr == VIS_SCENE_OBJECT)]
            for obj in sceneObjects:
                if self.key in obj.params.children:
                    params = CopyParams(obj.params)
                    params.children.remove(self.key)
                    negMsgHandler.internalRecvParams( obj.key, params )
                    negMsgHandler.sendParams( obj.key, params )


            self.sendDelete()

            theNet().remove(self.performerScene)

        VisItem.delete(self, isInitialized, negMsgHandler)


    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        realChange = ParamsDiff(self.params, params)

        VisItem.setParams(self, params, negMsgHandler, sendToCover)
        
        if 'filename' in realChange:
            self.parseCoxml()
            
        if sendToCover:
            if ('transX' in realChange) or \
               ('transY' in realChange) or \
               ('transZ' in realChange):
                self.sendObjMoved()
            if 'width' in realChange or 'height' in realChange or 'length' in realChange:
                self.sendGeometry()
            if 'variant_selected' in realChange:
                self.sendVariants()
            if 'appearance_colors' in realChange:
                self.sendAppearance()
            if 'kinematics_state' in realChange:
                self.sendKinematics();
            if len(self.params.children) > 0:
                self.sendChildren()


    def sendDelete(self):
        """ send delete msg to cover """
        if not self.covise_key=='No key':
            msg = coGRObjDelMsg( self.covise_key, 1)
            covise.sendRendMsg(msg.c_str())

    def sendObjMoved(self):
        """ send delete msg to cover """
        if not self.covise_key=='No key':
            msg = coGRObjMovedMsg( self.covise_key, float(self.params.transX), float(self.params.transY), float(self.params.transZ), float(self.params.rotX), float(self.params.rotY), float(self.params.rotZ), float(self.params.rotAngle))
            covise.sendRendMsg(msg.c_str())

    def sendGeometry(self):
        """ send delete msg to cover """
        if not self.covise_key=='No key':
            if (self.params.width == None):
                w=0.0
            else:
                w=self.params.width
            if (self.params.height == None):
                h=0.0
            else:
                h=self.params.height
            if (self.params.length == None):
                l=0.0
            else:
                l=self.params.length
            msg = coGRObjGeometryMsg( self.covise_key, w, h, l)
            covise.sendRendMsg(msg.c_str())

    def sendParent(self):
        """ send parent to plugin (if parent couldn't send child because of a missing covise key)"""
        if not self.covise_key=='No key':
            if self.parentCoviseKeyToSend != 'No key':
                msg = coGRObjAddChildMsg( self.parentCoviseKeyToSend, self.covise_key, 0)
                covise.sendRendMsg(msg.c_str()) 
                self.parentCoviseKeyToSend = 'No key'  
       

    def sendChildren(self):
        """ send list of children to plugin """
        if not self.covise_key=='No key':
            for childKey in self.params.children:
                covise_key_child = Neg2Gui.theNegMsgHandler().internalRequestObjectCoviseKey(childKey)
                if covise_key_child == 'No key':
                    child = globalKeyHandler().getObject(childKey)
                    if child:
                        child.parentCoviseKeyToSend = self.covise_key                  
                else:
                    msg = coGRObjAddChildMsg( self.covise_key, covise_key_child, 0)
                    covise.sendRendMsg(msg.c_str())


    def sendVariants(self):
        """ send variant msg to cover """
        if not self.covise_key=='No key':
            for groupName, variantName in iter(self.params.variant_selected.items()):
                msg = coGRObjSetVariantMsg( self.covise_key, str(groupName), str(variantName))
                covise.sendRendMsg(msg.c_str())

    def sendAppearance(self):
        """ send appearance msg to cover """
        if not self.covise_key=='No key':
            for scopeName, color in iter(self.params.appearance_colors.items()):
                coverColor = [float(c)/255.0 for c in list(color)]
                msg = coGRObjSetAppearanceMsg( self.covise_key, str(scopeName), *coverColor)
                covise.sendRendMsg(msg.c_str())
                
    def sendKinematics(self):
        msg = coGRObjKinematicsStateMsg( self.covise_key, self.params.kinematics_state)
        covise.sendRendMsg(msg.c_str())

    def parseCoxml(self):
        # open coxml
        dom = xml.dom.minidom.parse(covise.getCoConfigEntry("vr-prepare.Coxml.ResourceDirectory") + "/coxml/" + self.params.filename)

        # Class
        classElems = dom.getElementsByTagName("class")
        if len(classElems) > 0:
            self.params.classname = classElems[0].getAttribute("value")

        # Classification
        classificationElems = dom.getElementsByTagName("classification")
        if len(classificationElems) > 0:
            classificationElem = classificationElems[0]
            # name
            elems = classificationElem.getElementsByTagName("name")
            if len(elems) > 0:
                self.params.name = unicode(elems[0].getAttribute("value"))
            # product_line
            elems = classificationElem.getElementsByTagName("product_line")
            if len(elems) > 0:
                self.params.product_line = unicode(elems[0].getAttribute("value"))
            # model
            elems = classificationElem.getElementsByTagName("model")
            if len(elems) > 0:
                self.params.model = unicode(elems[0].getAttribute("value"))
            # description
            elems = classificationElem.getElementsByTagName("description")
            if (len(elems) > 0) and (elems[0].firstChild != None) and (elems[0].firstChild.nodeType == xml.dom.Node.TEXT_NODE):
                self.params.description = unicode(elems[0].firstChild.data)

        # Behaviors
        behaviorElems = dom.getElementsByTagName("behavior")
        if len(behaviorElems) > 0:
            for behaviorElem in behaviorElems[0].childNodes:
                if behaviorElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                    self.params.behaviors.append(behaviorElem.nodeName)
                    # VariantBehavior
                    if (behaviorElem.nodeName == "VariantBehavior"):
                        for groupElem in behaviorElem.childNodes:
                            if groupElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                                # group
                                groupName = groupElem.getAttribute("name")
                                variants = []
                                for variantElem in groupElem.childNodes:
                                    if variantElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                                        #variant
                                        variantName = variantElem.getAttribute("name")
                                        variants.append(variantName)
                                        if (groupName not in self.params.variant_selected):
                                            self.params.variant_selected[groupName] = variantName
                                self.params.variant_groups[groupName] = variants
                    # AppearanceBehavior
                    if (behaviorElem.nodeName == "AppearanceBehavior"):
                        for scopeElem in behaviorElem.childNodes:
                            if scopeElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                                # scope
                                scopeName = scopeElem.getAttribute("name")
                                # search color and palette
                                for colorElem in scopeElem.childNodes:
                                    if colorElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE and colorElem.nodeName == "color":
                                        # color
                                        values = colorElem.getAttribute("value")
                                        color = tuple([int(float(v)*255.0) for v in values.split()])
                                        if (len(color) == 4):
                                            self.params.appearance_colors[scopeName] = color
                                    elif colorElem.nodeType == xml.dom.minidom.Node.ELEMENT_NODE and colorElem.nodeName == "palette":
                                        # palette
                                        value = str(colorElem.getAttribute("value"))
                                        self.params.appearance_palettes[scopeName] = value

        # Misc
        geometryElems = dom.getElementsByTagName("geometry")
        if len(geometryElems) > 0:
            geometryElem = geometryElems[0]
            # width
            elems = geometryElem.getElementsByTagName("width")
            if len(elems) > 0:
                try:
                    self.params.width = float(elems[0].getAttribute("value"))
                except:
                    self.params.width = None
            # height
            elems = geometryElem.getElementsByTagName("height")
            if len(elems) > 0:
                try:
                    self.params.height = float(elems[0].getAttribute("value"))
                except:
                    self.params.height = None
            # length
            elems = geometryElem.getElementsByTagName("length")
            if len(elems) > 0:
                try:
                    self.params.length = float(elems[0].getAttribute("value"))
                except:
                    self.params.length = None


class SceneObjectVisParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name = 'SceneObjectVisParams'
        SceneObjectVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'filename' : '',
            'classname' : '',
            'children' : [],
            # Coxml Classification
            'product_line' : '',
            'model' : '',
            'description' : '',
            # Coxml Behaviors
            'behaviors' : [],
            'variant_groups' : {}, # groupname -> [variant1, variant2, ...]
            'variant_selected' : {}, # groupname -> activeVariant
            'appearance_palettes' : {}, # scopename -> palette
            'appearance_colors' : {}, # scopename -> (color Tuple RGBA (range 0 - 255)) 
            'kinematics_state' : '',
            # Coxml Misc
            'width' : None,
            'height' : None,
            'length' : None,
            # Transformation
            'transX' : 0.0,
            'transY' : 0.0,
            'transZ' : 0.0,
            'rotX' : 0.0,
            'rotY' : 0.0,
            'rotZ' : 0.0,
            'rotAngle' : 1.0,
            # Performer Scene Params
            'scale' : -1.0,
            'backface' : False,
            'orientation_iv' : False,
            'convert_xforms_iv' : False
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        if VisItemParams.isStaticParam(self, paramname):
            return True
        return paramname in ["filename"]
