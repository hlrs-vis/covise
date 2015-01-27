
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

# definition of classes ImportModule, Import2DModule,Import3DModule
import os
import os.path
import covise
from VRPCoviseNetAccess import saveExecute, connect, disconnect, theNet, ConnectionPoint, RWCoviseModule#, TransformModule

from coPyModules import BoundingBox, MakeOctTree, Colors, RWCovise
from BoundingBox import BoundingBoxParser, Box
import coviseStartup
from coviseCase import SCALARVARIABLE
from Utils import getExistingFilename

from ErrorManager import CoviseFileNotFoundError

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

enableCachedOctTrees = False
if covise.coConfigIsOn("vr-prepare.CachedOctTrees"):
    enableCachedOctTrees = True    

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    import a part( 2d or 3d geometry) and all data on that part """
"""                                                                """
"""    load files only once                                        """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class ImportModule(object):

    def __init__(self, dimension, partcase ):
        global testPath
        self._dim = dimension
        self._name = partcase.name
        self._part = partcase

        # Module Classes holding geo, oct and data
        self._geo = None
        self._data = {}
        self._oct = None

        # list of loaded files
        self._files_loaded = []
        self._oct_ready = False
        self._octTreeFileName = None
        self._octTreeFileExists = False
        self._octTreeWriter = None      # RWCovise to write out octtree

        # mapping variable name to filename and filetype
        self._dataFileNames = {}
        self._dataVariableType = {}
        self.__dataConnectionPoints = {}

        # bounding box module and content
        self._bb = None
        self._boundingBox = None
        self._numTimeSteps = None

        # bounding box module and content from unfiltered geometry
        self.__bbFromGeoRWCovise = None              # the module
        self.__boundingBoxFromGeoRWCovise = None     # the AABB

        # colors module to calculate min/max
        self._minMax = None

        
        
    def readPartcase(self):
        varNotFound = []
        for v in self._part.variables:
            if getExistingFilename(v.filename) == None:
                self._part.variables.remove(v)
                varNotFound.append( v.filename )
            else:
                self._dataFileNames[ v.name ] = v.filename
                self._dataVariableType[ v.name ] = v.variableDimension
        # check if file is transient
        self._isTransient = False
        filename = getExistingFilename(self._part.filename)
        if filename == None:
            raise CoviseFileNotFoundError(self._part.filename)
        in_file = open(filename, "rb") # open in binary mode (makes a difference on windows (read() may stop too early))
        # first check if we have TIMESTEP at the end
        in_file.seek(-100, os.SEEK_END)
        tail = in_file.read(100)
        if b"TIMESTEP" in tail:
            self._isTransient = True
        else:
            # if not already recognized as transient, check if we have SETELE at the beginning
            head = in_file.read(100)
            if b"SETELE" in head:
                # if we have, check the entire file since we might have nested sets
                in_file.seek(0, os.SEEK_BEGIN)
                line = in_file.readline()
                while line:
                    if b"TIMESTEP" in line:
                        self._isTransient = True
                        break
                    line = in_file.readline()
        in_file.close()
        return varNotFound
        
        
    """ ------------------------ """
    """ init by starting modules """
    """ ------------------------ """

    def _initGeo(self):
        if self._geo==None:
            self._geo = RWCoviseModule(self._part.filename)
            self._part.filename = self._geo.gridPath()

    def _initData(self, name):
        if name in self._dataFileNames:
            if not name in self._data:
                self._data[name] = RWCoviseModule( self._dataFileNames[name] )
                self._dataFileNames[name] = self._data[name].gridPath()
                self.__dataConnectionPoints[name] = self._data[name].connectionPoint()

    def _initOct(self):
        self._initGeo()
        if self._oct==None:
            # create or use disk-cached octtrees
            if enableCachedOctTrees == True and self._dim == 3:
                basename, extension = os.path.splitext(self._part.filename)
                self._octTreeFileName = basename + ".octtree" + extension
                print("self._octTreeFileName = ", self._octTreeFileName)
                if os.path.isfile(self._octTreeFileName) == False:
                    # create disk-cached octtree
                    self._octTreeFileExists = False
                    self._oct = MakeOctTree()
                    theNet().add(self._oct)
                    self._octIn  = ConnectionPoint( self._oct, 'inGrid' )
                    self._octOut = ConnectionPoint( self._oct, 'outOctTree' )
                    connect( self.geoConnectionPoint(), self._octIn )
                    
                    # connect RWCovise to MakeOctTree
                    self._octTreeWriter = RWCovise()       # writable
                    theNet().add(self._octTreeWriter)
                    self._octTreeWriter.set_grid_path(self._octTreeFileName)
                    connect(self._octOut, ConnectionPoint(self._octTreeWriter, 'mesh_in'))
                else:
                    # use disk-cached octtree
                    self._octTreeFileExists = True
                    self._oct = RWCovise()
                    theNet().add(self._oct)
                    self._oct.set_grid_path(self._octTreeFileName)
                    # cached octtrees must never get an input connection (RWCovise!)
                    self._octIn = None # ConnectionPoint(self._oct, 'mesh_in')
                    self._octOut = ConnectionPoint(self._oct, 'mesh')
            else:
                self._oct = MakeOctTree()
                theNet().add(self._oct)
                self._octIn  = ConnectionPoint( self._oct, 'inGrid' )
                self._octOut = ConnectionPoint( self._oct, 'outOctTree' )
                connect( self.geoConnectionPoint(), self._octIn )
        else:
            if enableCachedOctTrees == True and self._octTreeFileExists == True:
                # no reconnect necessary, if using disk-cached octtree
                pass
            else:
                # reconnect OctTree
                theNet().disconnectAllFromModulePort(self._oct, 'inGrid')
                connect(self.geoConnectionPoint(), self._octIn)


    """ ------------------------ """
    """ connection points        """
    """ ------------------------ """

    def geoConnectionPoint(self):
#        print("ImportModule::geoConnectionPoint() called")
        self._initGeo()
        return self._geo.connectionPoint()

    def dataConnectionPoint(self, name ):
        if name in self._dataFileNames:
            self._initData(name)
            #return self._data[name].connectionPoint()
            return self.__dataConnectionPoints[name]
        return None

    def octTreeConnectionPoint(self):
        self._initOct()
        return self._octOut

    def boundingBoxConnectionPoint(self):
        self.getBox()
        return ConnectionPoint(self._bb, 'GridOut0' )


    """ ------------------------ """
    """ execution methods        """
    """ ------------------------ """

    def execute(self):
        self.executeGeo()
        self.executeOct()
        for name in self._data:
            self.executeData(name)

    def executeGeo(self):
        self._initGeo()
        if not self._part.filename in self._files_loaded:
            _infoer.function = str(self.executeGeo)
            _infoer.write("Loading file " + self._part.filename)
            self._geo.execute()
            self._files_loaded.append(self._part.filename)
            return True
        return False

    def executeData( self, name ):
#        print("ImportModule::executeData() called")
        _infoer.function = str(self.executeGeo)
        _infoer.write("Load request for  " + name)
        if name in self._dataFileNames:
            self._initData(name)
            if not self._dataFileNames[name] in self._files_loaded:
                _infoer.write("Loading  " + self._dataFileNames[name])
                self._data[name].execute()
                self._files_loaded.append(self._dataFileNames[name])
                return True
        else :                        
#            print("Import Module: no variable called %s in part %s " % ( name,  self._name ))
            assert False
        _infoer.write("Returning False")
        return False

    def executeOct(self):
        self._initOct()
        if not self._oct_ready:
            if enableCachedOctTrees == True and self._octTreeFileExists == True:
                # cached octtrees aren't connected to the geo-RWCovise
                saveExecute(self._oct)
            elif not self.executeGeo():
                saveExecute(self._oct)
            self._oct_ready = True
            return True
        return False

    def reloadGeo(self):
        if not self.executeGeo():
            self._geo.execute()

    """ ------------------------ """
    """ delete                   """
    """ ------------------------ """

    def delete(self):
        if hasattr(self, "_geo") and self._geo: self._geo.remove()
        if hasattr(self, "_data"):
            for module in self._data.values(): module.remove()
        if hasattr(self, "_oct") and self._oct: theNet().remove(self._oct)
        if hasattr(self, "_octTreeWriter") and self._octTreeWriter: theNet().remove(self._octTreeWriter)
        if hasattr(self, "_bb") and self._bb: theNet().remove(self._bb)
        if hasattr(self, "_ImportModule__bbFromGeoRWCovise") and self.__bbFromGeoRWCovise: theNet().remove(self.__bbFromGeoRWCovise)
        if hasattr(self, "_minMax") and self._minMax: theNet().remove(self._minMax)


    """ ------------------------ """
    """ read private variables   """
    """ ------------------------ """

    def getDimension(self):
        return self._dim
    def getName(self):
        return self._name

    def getParts(self):
        return [self._part]
    def getPartCase(self):
        return self._part


    def getBoxFromGeoRWCovise(self):
        """ return the bounding box from the originally unfiltered geometry """

        if self.__boundingBoxFromGeoRWCovise == None:
            self.__bbFromGeoRWCovise = BoundingBox()
            theNet().add(self.__bbFromGeoRWCovise)
            connect( self._geo.connectionPoint(), ConnectionPoint(self.__bbFromGeoRWCovise, 'GridIn0'))

            # Clear info queue so we dont read a previous BB output.
            # (If something goes wrong with the queue, this could be the reason.)
            coviseStartup.globalReceiverThread.infoQueue_.clear()
            saveExecute(self.__bbFromGeoRWCovise)
            boxParser = BoundingBoxParser()
            boxParser.parseQueue(coviseStartup.globalReceiverThread.infoQueue_)
            self.__boundingBoxFromGeoRWCovise = boxParser.getBox()

        return self.__boundingBoxFromGeoRWCovise

    def getBox(self, execute = False):
        """ return the bounding box """
        if self._bb==None:
            self._bb = BoundingBox()
            theNet().add( self._bb)
            connect( self.geoConnectionPoint(), ConnectionPoint( self._bb, 'GridIn0' ) )
            # Clear info queue so we dont read a previous BB output.
            # (If something goes wrong with the queue, this could be the reason.)
            coviseStartup.globalReceiverThread.infoQueue_.clear()
            if not self.executeGeo():
                saveExecute(self._bb)
            boxParser = BoundingBoxParser()
            boxParser.parseQueue(coviseStartup.globalReceiverThread.infoQueue_)
            self._boundingBox = boxParser.getBox()
            self._numTimeSteps = boxParser.getNumTimeSteps()
        elif execute:
            theNet().disconnectAllFromModule(self._bb)
            connect( self.geoConnectionPoint(), ConnectionPoint( self._bb, 'GridIn0' ) )
            # Clear info queue so we dont read a previous BB output.
            # (If something goes wrong with the queue, this could be the reason.)
            coviseStartup.globalReceiverThread.infoQueue_.clear()
            if not self.executeGeo():
                saveExecute(self._bb)
            boxParser = BoundingBoxParser()
            boxParser.parseQueue(coviseStartup.globalReceiverThread.infoQueue_)
            try:
                oldbb = self._boundingBox
                self._boundingBox = boxParser.getBox()
                #self._numTimeSteps = boxParser.getNumTimeSteps()
            except (ValueError):
                self._boundingBox = oldbb

        if self._bb != None:
            theNet().disconnectAllFromModulePort(self._bb, 'GridIn0')
            connect( self.geoConnectionPoint(), ConnectionPoint( self._bb, 'GridIn0' ) )

        return self._boundingBox
        
    def getNumTimeSteps(self):
        if not self._numTimeSteps:
            self.getBox()
        return self._numTimeSteps
    

    def getDataMinMax( self, variable):
        """ return min and max value of variable """
        if variable==None:
            return

        if self._minMax==None:
            self._minMax = Colors()
            theNet().add(self._minMax)
        theNet().disconnectAllFromModulePort( self._minMax, 'DataIn0' )
        connect(self.dataConnectionPoint(variable), ConnectionPoint(self._minMax, 'DataIn0'))

        if not self.executeData( variable ):
            saveExecute(self._minMax)
            
        return ( float(self._minMax.getParamValue('MinMax')[0]),\
                 float(self._minMax.getParamValue('MinMax')[1]) )

    def getIsTransient(self):
        return self._isTransient

    """ ------------------------ """
    """ return status string     """
    """ ------------------------ """
    def __str__(self):
        string = 'Status of ' + self._name + '\n'
        for v in self._part.variables:
            string = string + v.name + '\n'
        return string

    def getCoObjName(self):
        if not self._geo==None:
            return self._geo.getCoObjName()


""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                            """
"""    dimension specific import modules                       """
"""                                                            """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class Import2DModule( ImportModule ):

    def __init__(self, partcase ):
        ImportModule.__init__(self, 2, partcase )

class Import3DModule( ImportModule ):

    def __init__(self, partcase ):
        ImportModule.__init__(self, 3, partcase )

