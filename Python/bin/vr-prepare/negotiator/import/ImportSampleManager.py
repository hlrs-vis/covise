
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

# definition of classes ImportSample3DModule, ImportSample2DModule

#from ImportManager import Import3DModule, Import2DModule
from ImportTransformManager import ImportTransform2DModule, ImportTransform3DModule

from VRPCoviseNetAccess import saveExecute, connect, disconnect, theNet, ConnectionPoint, RWCoviseModule, BlockCollectModule, AssembleUsgModule
import covise

from coPyModules import BoundingBox, MakeOctTree, Colors, Collect, Sample, ShowGrid
from BoundingBox import BoundingBoxParser, Box
import coviseStartup


""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    import module, but Sample 3d grid                           """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

USER_DEFINED = 2
MAX_FLT = 1

ParentClass3D = ImportTransform3DModule

#class ImportSample3DModule(Import3DModule):
class ImportSample3DModule(ParentClass3D):
    def __init__(self, partcase):
#        Import3DModule.__init__(self, partcase)
        ParentClass3D.__init__(self, partcase)
        #series of sample modules. Referenced by outside type
        self._sample = {}
        self._bbox = {}      
        
    def _initSample( self, varname, outside ):
        if not (varname,outside) in self._sample:
            self._sample[(varname,outside)] = Sample()
            theNet().add(self._sample[(varname,outside)])
            self._bbox[(varname,USER_DEFINED)] = Box()
            self._bbox[(varname,MAX_FLT)] = Box()

#            connect( Import3DModule.geoConnectionPoint(self), ConnectionPoint(self._sample[(varname,outside)], 'GridIn') )
#            connect( Import3DModule.dataConnectionPoint(self, varname), ConnectionPoint(self._sample[(varname,outside)], 'DataIn') )
            connect( ParentClass3D.geoConnectionPoint(self), ConnectionPoint(self._sample[(varname,outside)], 'GridIn') )
            connect( ParentClass3D.dataConnectionPoint(self, varname), ConnectionPoint(self._sample[(varname,outside)], 'DataIn') )

            sample = self._sample[(varname,outside)]
            sample.set_outside(outside)
            if not ParentClass3D.executeOct(self) and not ParentClass3D.executeData(self, varname):
                saveExecute(sample)

        # reconnect Sample
#        print "RECONNECTING 3D SAMPLE"
        theNet().disconnectAllFromModulePort(self._sample[(varname,outside)], 'GridIn')
        theNet().disconnectAllFromModulePort(self._sample[(varname,outside)], 'DataIn')
        connect( ParentClass3D.geoConnectionPoint(self), ConnectionPoint(self._sample[(varname,outside)], 'GridIn') )
        connect( ParentClass3D.dataConnectionPoint(self, varname), ConnectionPoint(self._sample[(varname,outside)], 'DataIn') )

    def _initSampleUser(self, varname):
        self._initSample(varname, USER_DEFINED)

    def _initSampleMax(self, varname):
        self._initSample(varname, MAX_FLT)

    def _updateSample( self, varname, bbox, outside ):
        self._initSample(varname, outside)
        if not bbox==self._bbox[(varname,outside)]:
            self._sample[(varname,outside)].set_P1_bounds( bbox.getXMin(), bbox.getYMin(), bbox.getZMin() )
            self._sample[(varname,outside)].set_P2_bounds( bbox.getXMax(), bbox.getYMax(), bbox.getZMax() )
            self._sample[(varname,outside)].set_bounding_box(2)
            self._bbox[(varname,outside)] = bbox                

    def geoSampleConnectionPoint(self, varname=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._initSample( varname, outside)
        return ConnectionPoint(self._sample[(varname,outside)], 'GridOut')        

    def dataSampleConnectionPoint(self, varname=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._initSample( varname, outside)
        return ConnectionPoint(self._sample[(varname,outside)], 'DataOut')

    def executeSampleData(self, varname, bbox=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        if bbox==None:
            if not (varname,outside) in self._bbox:
                self._bbox[(varname,outside)] = Box()
            box = self._bbox[(varname,outside)]
        else:
            box = bbox
        self._updateSample( varname, box, outside)                   
        sample = self._sample[(varname,outside)]

        saveExecute(sample)

    def delete(self):
        if hasattr(self, "_sample"):
            for module in self._sample.values(): theNet().remove(module)
        ParentClass3D.delete(self)

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    import module, but Sample polygons                          """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

ParentClass2D = ImportTransform2DModule

#class ImportSample2DModule(Import2DModule):
class ImportSample2DModule(ParentClass2D):
    def __init__(self, partcase):
#        Import2DModule.__init__(self, partcase)
        ParentClass2D.__init__(self, partcase)
        self._sample = None
        self._showgrid = None

    def _initSample(self, varname=None, outside=USER_DEFINED):
        if self._sample==None:
            self._sample = Sample()
            theNet().add(self._sample)
#            connect( Import2DModule.geoConnectionPoint(self), ConnectionPoint(self._sample, 'GridIn') )
            connect( ParentClass2D.geoConnectionPoint(self), ConnectionPoint(self._sample, 'GridIn') )

            self._showgrid = ShowGrid()
            theNet().add(self._showgrid)
            connect( ConnectionPoint(self._sample, 'GridOut'), ConnectionPoint(self._showgrid, 'meshIn') )

            sample = self._sample
            if not ParentClass2D.executeOct(self) and not ParentClass2D.executeData(self, varname):
                saveExecute(sample)


        # reconnect Sample
#        print "RECONNECTING 2D SAMPLE"
        theNet().disconnectAllFromModulePort(self._sample, 'GridIn')
        connect( ParentClass2D.geoConnectionPoint(self), ConnectionPoint(self._sample, 'GridIn') )


    def _updateSample( self, dim, varname=None, bbox=None, outside=USER_DEFINED ):
    #def _updateSample( self, dim ):
        self._initSample(varname, outside)
        #update dimension
        self._sample.set_isize(1)
        self._sample.set_user_defined_isize(dim)
        self._sample.set_jsize(1)
        self._sample.set_user_defined_jsize(dim)
        self._sample.set_ksize(1)
        self._sample.set_user_defined_ksize(dim)

    def geoSampleConnectionPoint(self, varname=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._initSample(varname, outside)
        return ConnectionPoint(self._showgrid, 'points')

    def dataSampleConnectionPoint(self, varname=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._initSample(varname, outside)
        return ConnectionPoint(self._sample, 'DataOut')

    def executeSampleGeo( self, dim ):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._updateSample( dim )
#        if not Import2DModule.executeGeo(self):
        if not ParentClass2D.executeGeo(self):
            saveExecute(self._sample)

    def executeSampleData(self, varname, bbox=None, outside=USER_DEFINED):
        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            return

        self._updateSample(2, varname, bbox, outside)
        sample = self._sample

        saveExecute(sample)

    def delete(self):
        if hasattr(self, "_sample"):
            if self._sample: theNet().remove(self._sample)
        ParentClass2D.delete(self)
