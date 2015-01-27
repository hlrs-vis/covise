
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

# definition of classes ImportGroupModule, ImportGroup2DModule, ImportGroup3DModule, ImportSimpleGroupModule, ImportSimpleGroup2DModule, ImportSimpleGroup3DModule


from ImportManager import ImportModule
from ImportSampleManager import USER_DEFINED, MAX_FLT
from VRPCoviseNetAccess import theNet,saveExecute, connect, ConnectionPoint, RWCoviseModule, BlockCollectModule, AssembleUsgModule

from BoundingBox import Box
import coviseStartup

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    group of import modules                                     """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

COMPOSED_VELOCITY = 'Composed Velocity'

class ImportGroupModule(object):
    def __init__(self, dimension, isTransient=False):
        self._dim = dimension
        self._parts = []

        self._isTransient = isTransient
        
        #series of block collect modules
        self._collectGeo = []
        # list of block collects for dat. Referenced by variable name
        self._collectDataPerName = {}
        # series of block collect modules for octtree
        self._collectOct = []
        # list of block collects for sampled dat. Referenced by variable name
        self._collectSampleGeoPerName = {}
        self._collectSampleDataPerName = {}
        # local velocity value if "velocity" is requested
        self._velMapping = []
        self._firstTimeGeo = True
        self._firstTimeData = {}
        self._firstTimeSample = {}
        self._firstTimeOct = True

    def addImport( self, importModule, velValue=None ):
        # importModule is an ImportModule or ImportGroupModule class
        if importModule.getDimension()==self._dim :
            self._parts.append( importModule )
            self._velMapping.append(velValue)
            self._initGeo(importModule.geoConnectionPoint())

    def _initObj( self, bclist, connectionPoint, part_cnt):
        if len(bclist)==0:
            bc = BlockCollectModule(self.getIsTransient())
            bclist.append( bc )
            port = part_cnt
        elif part_cnt>=15:
            # number of block collect module
            nb = (part_cnt-15) / 14 + 1
            if nb==len(bclist):
                bc = BlockCollectModule(self.getIsTransient())
                bclist.append( bc )
                connect( bclist[ len(bclist)-2].objOutConnectionPoint(), bc.objInConnectionPoint(0) )
            else:
                bc =  bclist[nb]
            port = (part_cnt-15) % 14 + 1
        else:
            bc = bclist[0]
            port = part_cnt
        connect( connectionPoint, bc.objInConnectionPoint(port) )

    def _initGeo( self, geoConnectionPoint ):
        self._initObj( self._collectGeo, geoConnectionPoint, len(self._parts) - 1)

    def _varInPart( self, vname, part_cnt):
        """ return name of variable in part part_cnt """
        if vname==COMPOSED_VELOCITY:
            varname = self._velMapping[part_cnt]
        else:
            varname = vname
        return varname

    def _initData( self, vname ):
        if not vname in self._collectDataPerName :
            var_bclist = []
            part_cnt=0
            for part in self._parts:
                varname = self._varInPart( vname, part_cnt)
                self._initObj( var_bclist, part.dataConnectionPoint(varname), part_cnt )
                part_cnt = part_cnt + 1
            self._collectDataPerName[vname] = var_bclist
            return True
        else:
            return False

    def _initSample( self, vname, outside=2 ):
        if not (vname,outside) in self._collectSampleDataPerName:
            geo_var_bclist = []
            data_var_bclist = []
            part_cnt=0
            for part in self._parts:
                varname = self._varInPart( vname, part_cnt)
                self._initObj( geo_var_bclist, part.geoSampleConnectionPoint(varname, outside), part_cnt )
                self._initObj( data_var_bclist, part.dataSampleConnectionPoint(varname, outside), part_cnt )
                part_cnt = part_cnt + 1
            self._collectSampleGeoPerName[(vname,outside)] = geo_var_bclist
            self._collectSampleDataPerName[(vname,outside)] = data_var_bclist
            return True
        else:
            return False

    def _initOct(self, octTreeConnectionPoint, part_cnt):
        self._initObj( self._collectOct, octTreeConnectionPoint, part_cnt)

    """ ------------------------ """
    """ delete                   """
    """ ------------------------ """

    def delete(self):
        if hasattr(self, "_collectGeo"):
            for module in self._collectGeo: module.remove()
        if hasattr(self, "_collectDataPerName"):
            for l in self._collectDataPerName.values():
                for module in l: module.remove()
        if hasattr(self, "_collectOct"):
            for module in self._collectOct: module.remove()
        if hasattr(self, "_collectSampleGeoPerName"):
            for l in self._collectSampleGeoPerName.values():
                for module in l: module.remove()
        if hasattr(self, "_collectSampleDataPerName"):
            for l in self._collectSampleDataPerName.values():
                for module in l: module.remove()

    """ ------------------------ """
    """ connection points        """
    """ ------------------------ """

    def geoConnectionPoint(self):
        if len(self._collectGeo)!=0:
            return self._collectGeo[len(self._collectGeo)-1].objOutConnectionPoint()

    def dataConnectionPoint( self, vname ):
        self._initData( vname )
        var_bclist = self._collectDataPerName[vname]
        return var_bclist[len(var_bclist)-1].objOutConnectionPoint()

    def octTreeConnectionPoint( self ):
        if len(self._collectOct)==0:
            part_cnt=0
            for part in self._parts:
                self._initOct( part.octTreeConnectionPoint(), part_cnt )
                part_cnt += 1
        return self._collectOct[len(self._collectOct)-1].objOutConnectionPoint()

    def geoSampleConnectionPoint(self, varname, outside=2):
        self._initSample( varname, outside )
        var_bclist = self._collectSampleGeoPerName[(varname,outside)]
        return var_bclist[len(var_bclist)-1].objOutConnectionPoint()

    def dataSampleConnectionPoint(self, varname, outside=2):
        self._initSample( varname, outside )
        var_bclist = self._collectSampleDataPerName[(varname,outside)]
        return var_bclist[len(var_bclist)-1].objOutConnectionPoint()

    """ ------------------------ """
    """ execution methods        """
    """ ------------------------ """

    def executeGeo(self):
        one_executed = False

        # reconnecting
        numPorts = 0
        numModules = 0
        for part in self._parts:
            # dont reconnect BlockCollect inter-connections
            if (numPorts % 15 == 0) and (numModules != 0):
                numPorts += 1
            theNet().disconnectAllFromModulePort(self._collectGeo[numModules].objInConnectionPoint(numPorts).module, self._collectGeo[numModules].objInConnectionPoint(numPorts).port)
            theNet().connect(part.geoConnectionPoint().module, part.geoConnectionPoint().port, self._collectGeo[numModules].objInConnectionPoint(numPorts).module, self._collectGeo[numModules].objInConnectionPoint(numPorts).port)

            numPorts = numPorts + 1
            if numPorts % 15 == 0:
                numModules = numModules +1
                numPorts = 0

        numPorts = 0
        numModules = 0
        for part in self._parts:
            #theNet().disconnectAllFromModulePort(self._collectGeo[numModules].objInConnectionPoint(numPorts).module, self._collectGeo[numModules].objInConnectionPoint(numPorts).port)
            #theNet().connect(part.geoConnectionPoint().module, part.geoConnectionPoint().port, self._collectGeo[numModules].objInConnectionPoint(numPorts).module, self._collectGeo[numModules].objInConnectionPoint(numPorts).port)

            if part.executeGeo():
                one_executed = True
            numPorts = numPorts + 1
            if numPorts % 15 == 0:
                numModules = numModules +1
                numPorts = 0
        if not one_executed and self._firstTimeGeo:
            for bc in self._collectGeo:
                bc.execute()
                one_executed = True
        self._firstTimeGeo = False
        return one_executed

    def executeData(self, vname):
        one_executed = False
        initialized = self._initData(vname)
        collectData = self._collectDataPerName[vname]

        # reconnecting
        part_cnt=0
        numModules = 0
        numPorts = 0
        for part in self._parts:
            varname = self._varInPart( vname, part_cnt)
            # dont reconnect BlockCollect inter-connections
            if (numPorts % 15 == 0) and (numModules != 0):
                numPorts += 1
            theNet().disconnectAllFromModulePort(collectData[numModules].objInConnectionPoint(numPorts%15).module, collectData[numModules].objInConnectionPoint(numPorts%15).port)
            theNet().connect(part.dataConnectionPoint(varname).module, part.dataConnectionPoint(varname).port,
                             collectData[numModules].objInConnectionPoint(numPorts%15).module, collectData[numModules].objInConnectionPoint(numPorts%15).port)
            part_cnt += 1
            numPorts += 1
            if numPorts % 15 == 0:
                numModules = numModules + 1


        part_cnt=0
        numModules = 0
        for part in self._parts:
            varname = self._varInPart( vname, part_cnt)

            part_cnt += 1
            if part_cnt % 15 == 0:
                numModules = numModules + 1
            _infoer.function = str(self.executeData)
            _infoer.write("Loading file " + varname)
            if part.executeData(varname):
                one_executed = True
        if not one_executed and self._firstTimeData.get(vname, True):
            for bc in self._collectDataPerName[vname]:
                bc.execute()
                one_executed = True
        self._firstTimeData[vname] = False
        return one_executed

    def executeSampleData(self, vname, bbox=None, outside=USER_DEFINED ):
        one_executed = False
        initialized = self._initSample( vname, outside )
        part_cnt=0
        numModules = 0
        numPorts = 0
        collectSampleData = self._collectSampleDataPerName[(vname,outside)]
        collectSampleGeo = self._collectSampleGeoPerName[(vname,outside)]
        for part in self._parts:
            varname = self._varInPart( vname, part_cnt)
            # dont reconnect BlockCollect inter-connections
            if (numPorts % 15 == 0) and (numModules != 0):
                numPorts += 1
            theNet().disconnectAllFromModulePort(collectSampleData[numModules].objInConnectionPoint(numPorts%15).module, collectSampleData[numModules].objInConnectionPoint(numPorts%15).port)
            theNet().connect(part.dataSampleConnectionPoint(varname,outside).module, part.dataSampleConnectionPoint(varname,outside).port,
                    collectSampleData[numModules].objInConnectionPoint(numPorts%15).module, collectSampleData[numModules].objInConnectionPoint(numPorts%15).port)
            theNet().disconnectAllFromModulePort(collectSampleGeo[numModules].objInConnectionPoint(numPorts%15).module, collectSampleGeo[numModules].objInConnectionPoint(numPorts%15).port)
            theNet().connect(part.geoSampleConnectionPoint(varname,outside).module, part.geoSampleConnectionPoint(varname,outside).port,
                             collectSampleGeo[numModules].objInConnectionPoint(numPorts%15).module, collectSampleGeo[numModules].objInConnectionPoint(numPorts%15).port)
            part_cnt+=1
            numPorts += 1
            if numPorts % 15 == 0:
                numModules = numModules + 1

        part_cnt = 0
        numModules = 0
        for part in self._parts:
            varname = self._varInPart( vname, part_cnt)
            part_cnt+=1
            if part_cnt % 15 == 0:
                numModules = numModules + 1
            if part.executeSampleData(varname, bbox, outside):
                one_executed = True
        if not one_executed and self._firstTimeSample.get(vname, True):
            for i in range(len(self._collectSampleGeoPerName[(vname,outside)])):
                self._collectSampleGeoPerName[(vname,outside)][i].execute()
                self._collectSampleDataPerName[(vname,outside)][i].execute()
                one_executed = True
        self._firstTimeSample[vname] = False
        return one_executed


    def executeOct(self):
        self.octTreeConnectionPoint()
        one_executed = False
        #quickfix
        #save/loading of a 2d composed probing point does not work with this
        #for part in self._parts:
        #    if part.executeOct():
        #        one_executed = True
        numPorts = 0
        numModules = 0
        for part in self._parts:
            # dont reconnect BlockCollect inter-connections
            if (numPorts % 15 == 0) and (numModules != 0):
                numPorts += 1
            theNet().disconnectAllFromModulePort(self._collectOct[numModules].objInConnectionPoint(numPorts).module, self._collectOct[numModules].objInConnectionPoint(numPorts).port)
            connect(part.octTreeConnectionPoint(), self._collectOct[numModules].objInConnectionPoint(numPorts))
            numPorts = numPorts + 1
            if numPorts % 15 == 0:
                numModules = numModules +1
                numPorts = 0

            if part.executeOct():
                one_executed = True

        if not one_executed and self._firstTimeOct:
            for bc in self._collectOct:
                bc.execute()
                one_executed = True
        self._firstTimeOct = False
        return one_executed

    def reloadGeo(self):
        self.executeGeo()

    """ ------------------------ """
    """ read private variables   """
    """ ------------------------ """

    def getDimension(self):
        return self._dim

    def getParts(self):
        return self._parts

    def getCoObjName(self):
        if len(self._collectGeo)!=0:
            return self._collectGeo[len(self._collectGeo)-1].getCoObjName()

    def getIsTransient(self):
        for part in self._parts:
            if not part.getIsTransient():
                return False

        # all parts are transient
        return True

    def getBoxFromGeoRWCovise(self):
        box = Box( (0,0),(0,0),(0,0) )
        for part in self._parts:
            box = box + part.getBoxFromGeoRWCovise()
        return box

    def getBox(self, forceExecute=False):
        box = Box( (0,0),(0,0),(0,0) )
        for part in self._parts:
            box = box + part.getBox(forceExecute)
        return box

    def getDataMinMax( self, variable):
        """ return min and max value of variable """
        dmin = 1.0e+28
        dmax = -1.0e+28
        part_cnt=0
        for part in self._parts:
            if variable==COMPOSED_VELOCITY:
                varname = self._velMapping[part_cnt]
                part_cnt += 1
            else:
                varname = variable
            localrange = part.getDataMinMax(varname)
            if localrange:
                dmin = min(dmin, float(localrange[0]))
                dmax = max(dmax, float(localrange[1]))
        return (dmin,dmax)



""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                            """
"""    dimension specific ImportGroupModules                   """
"""                                                            """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class ImportGroup3DModule(ImportGroupModule):

    def __init__(self, isTransient=False):
        ImportGroupModule.__init__(self, 3, isTransient)

class ImportGroup2DModule(ImportGroupModule):

    def __init__(self, isTransient=False):
        ImportGroupModule.__init__(self, 2, isTransient)

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    group of import modules, but                                """
"""    merge objects with AssembleUsg                              """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class ImportSimpleGroupModule(ImportGroupModule):
    def __init__(self, dimension, isTransient=False):
        ImportGroupModule.__init__(self, dimension,isTransient)
        #series of assemble usg modules. Referenced by variable name
        self._assembleUsg = {}
        self._assembleGeo = None

    def _initAssembleGeo(self):
        if self._assembleGeo==None:
            self._assembleGeo = AssembleUsgModule()

    def _initAssemble(self, varname):
        if not varname in self._assembleUsg:
            self._assembleUsg[varname] = AssembleUsgModule()

    def geoConnectionPoint(self, varname=None):
        if varname==None:
            if self._assembleGeo==None:
                self._initAssembleGeo()
                connect( ImportGroupModule.geoConnectionPoint(self), self._assembleGeo.geoInConnectionPoint() )
            return  self._assembleGeo.geoOutConnectionPoint()
        else:
            self.dataConnectionPoint(varname)
            return self._assembleUsg[varname].geoOutConnectionPoint()

    def dataConnectionPoint(self, varname):
        if not varname in self._assembleUsg:
            self._initAssemble(varname)
            connect( ImportGroupModule.geoConnectionPoint(self), self._assembleUsg[varname].geoInConnectionPoint() )
            connect( ImportGroupModule.dataConnectionPoint(self,varname), self._assembleUsg[varname].dataInConnectionPoint() )
        return  self._assembleUsg[varname].dataOutConnectionPoint()

    def octConnectionPoint( self ):
        self._initOct()
        self._octOut = ConnectionPoint( self._oct, 'outOctTree' )
        connect( self.geoConnectionPoint(), ConnectionPoint( self._oct, 'inGrid' ) )
        return self._octOut

    def executeGeo(self):
        self.geoConnectionPoint()
        ImportGroupModule.executeGeo(self)

    def executeData(self, varname):
        self.geoConnectionPoint(varname)
        ImportGroupModule.executeGeo(self)
        ImportGroupModule.executeData(self, varname)

    def executeOct(self):
        self.octTreeConnectionPoint()
        saveExecute(self._oct)

    def delete(self):
        if hasattr(self, "_assembleGeo") and self._assembleGeo: self._assembleGeo.remove()
        if hasattr(self, "_assembleUsg"):
            for module in self._assembleUsg.values(): module.remove()
        ImportGroupModule.delete(self)

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                            """
"""    dimension specific ImportGroupModules                   """
"""                                                            """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class ImportSimpleGroup3DModule(ImportSimpleGroupModule):

    def __init__(self):
        ImportSimpleGroupModule.__init__(self, 3)

class ImportSimpleGroup2DModule(ImportSimpleGroupModule):

    def __init__(self):
        ImportSimpleGroupModule.__init__(self, 2)

