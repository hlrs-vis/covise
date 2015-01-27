
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
try:
    from coPyModules import AddAttribute
except:
    class AddAttribute(object):
        pass # TODO (FIX AddAttribute)
from KeydObject import VIS_2D_STATIC_COLOR
from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class Part2DStaticColorVis(VisItem):
    """ VisItem to color an object with an rgba color """
    def __init__(self):
        VisItem.__init__(self, VIS_2D_STATIC_COLOR, self.__class__.__name__)
        self.params = Part2DStaticColorVisParams()
        self.__addAttribute = None

    def __init(self):
        """ start COVISE Module and connect output to COVER """
        if self.__addAttribute==None :
            self.__addAttribute = AddAttribute()
            theNet().add(self.__addAttribute)
            self.__addAttribute.set_attrName( "COLOR" )
            addAttributeIn = ConnectionPoint( self.__addAttribute, 'inObject' )
            self.__addAttributeOut = ConnectionPoint( self.__addAttribute, 'outObject' )
            connect( self.importModule.geoConnectionPoint(), addAttributeIn )
            VisItem.connectToCover( self, self )

    def __update(self):
        """ do init if necessary; update module parameters """
        self.__init()
        colorStr = "%s %s %s %s" % ( self.params.r, self.params.g, self.params.b, self.params.a )
        _infoer.function = str(self.__update)
        _infoer.write(": updating to " + colorStr )
        self.__addAttribute.set_attrVal( colorStr )

    def connectionPoint(self):
        return self.__addAttributeOut

    def run(self, runmode):
        _infoer.function = str(self.run)
        _infoer.write("go")
        self.__update()
        if not self.importModule.executeGeo():
            saveExecute(self.__addAttribute)

class Part2DStaticColorVisParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name = 'Part2DStaticColorVisParams'
        self.r = 255
        self.g = 255
        self.b = 255
        self.a = 255
