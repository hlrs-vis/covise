
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
from coPyModules import Colors, Collect
from KeydObject import VIS_2D_SCALAR_COLOR
from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class Part2DScalarColorVis(VisItem):
    """ VisItem to color an object with an rgba color """
    def __init__(self):
        VisItem.__init__(self, VIS_2D_SCALAR_COLOR, self.__class__.__name__)
        self.params = Part2DScalarColorVisParams()
        self.__colors = None
        self.__collect = None
        # currently used input connection to colors module
        self.__lastDataConnection = None

    def __init(self):
        """ start COVISE modules and connect output to COVER """
        if self.__colors==None and self.__collect==None:
            self.__colors = Colors()
            theNet().add(self.__colors)
            self.__colorsIn = ConnectionPoint(self.__colors, 'GridIn0')

            self.__collect = Collect()
            theNet().add(self.__collect)
            self.__collectOut = ConnectionPoint(self.__collect, 'GeometryOut0')
            connect( self.importModule.geoConnectionPoint(), ConnectionPoint(self.__collect, 'GridIn0') )

            VisItem.connectToCover( self, self )

    def __update(self):
        """ do init if necessary; update module parameters """
        self.__init()

        # update input
        dataInConnect = self.importModule.dataConnectionPoint(self.params.variable)
        if not self.__lastDataConnection==None :
            disconnect( self.__lastDataConnection, self.__colorsIn )
        else :
            theNet().connect(self.__colors, 'TextureOut0', self.__collect, 'TextureIn0')

        if dataInConnect:
            connect( self.importModule.dataConnectionPoint(self.params.variable), self.__colorsIn )
        self.__lastDataConnection=self.importModule.dataConnectionPoint(self.params.variable)

        _infoer.function = str(self.__update)
        _infoer.write(": updating to variable " + self.params.variable )

        # update colormap settings
        self.__colors.set_numSteps(self.params.numSteps)
        #self.__colros.set_colormap(self.params.colorTable)

    def connectionPoint(self):
        return self.__collectOut

    def run(self, runmode):
        _infoer.function = str(self.run)
        _infoer.write("go")
        self.__update()
        self.importModule.executeGeo()
        if not self.importModule.executeData(self.params.variable):
            saveExecute(self.__colors)

class Part2DScalarColorVisParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name = 'Part2DScalarColorVisParams'
        self.variable    = 'T'
        self.numSteps    = 16
        self.colorTable  = 0
