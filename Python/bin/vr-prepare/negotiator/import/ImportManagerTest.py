import math

from ImportManager import ImportModule, Import2DModule, Import3DModule
from ImportSampleManager import ImportSample2DModule, ImportSample3DModule
from ImportGroupManager import ImportGroupModule, ImportGroup2DModule, ImportGroup3DModule, COMPOSED_VELOCITY, ImportSimpleGroupModule
from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from VRPCoviseNetAccess import saveExecute, connect, disconnect, theNet, ConnectionPoint, RWCoviseModule, BlockCollectModule, AssembleUsgModule

from coPyModules import BoundingBox, MakeOctTree, Colors, Collect, Sample, ShowGrid
from BoundingBox import BoundingBoxParser, Box
import coviseStartup


""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                            """
"""                         T E S T S                          """
"""              TEST for the ImportManager,                   """
"""          ImportGroupManager, VPRImportSampleManager        """
"""                                                            """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


# Start: covise --script <name of this file>
# (Environment-variable PYTHONPATH can be used to let
# python find the necessary modules including this
# one.)

from coviseModuleBase import globalMessageQueue
from coviseCase import *

def testCase():
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
    return coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)

def testComposedCase():
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/data/Kunden/Kaercher/demo/pumpe.cocase')
    return coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)

def testImportModule2():
    dsc = testCase()
    return Import2DModule( dsc.parts2d[12] )

def testImportModule1():
    dsc = testCase()
    print("Loading ",  dsc.parts2d[8].name)
    return Import2DModule( dsc.parts2d[8] )

def testSampleImportModule1():
    dsc = testCase()
    print("Loading ",  dsc.parts3d[0].name)
    return ImportSample3DModule( dsc.parts3d[0] )

def testLargeGroup():
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
    dsc = coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)
    gip = ImportGroup2DModule()
    for part in dsc.parts2d:
        ip = Import2DModule(part)
        print(str(ip))
        gip.addImport(ip)
    return gip

def testSmallSimpleGroup():
    gip = ImportSimpleGroup2DModule()
    gip.addImport( testImportModule1() )
    #gip.addImport( testImportModule2() )
    return gip

def testSampleGroup():
    dsc = testComposedCase()
    p1 = ImportSample3DModule( dsc.parts3d[0] )
    p2 = ImportSample3DModule( dsc.parts3d[1] )
    gip = ImportGroup3DModule()
    gip.addImport(p1, 'Vels')
    gip.addImport(p2, 'vel')
    return gip

def testLargeSimpleGroup():
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
    dsc = coviseCase2DimensionSeperatedCase(nameAndCase.case, nameAndCase.name, nameAndCase.pathToCaseFile)
    gip = ImportSimpleGroupModule(2)
    for part in dsc.parts2d:
        ip = Import2DModule(part)
        print(str(ip))
        gip.addImport(ip)
    return gip

def testGeo():
    ip = testImportModule1()
    box = ip.getBox()
    (x,y,z) = box.getCenter()
    if not (x,y,z)==(-0.041321500000000011, 0.0, 0.1766115):
        print("test failed" ,(x,y,z))
    else:
        print("OK")

def testSample():
    #ip = testSampleImportModule1()
    ip = testSampleGroup()
    ip.executeGeo()
    #ip.executeOct()
    ip.executeSampleData(COMPOSED_VELOCITY)
    #(a,b) = ip.getDataMinMax('rotvel')#COMPOSED_VELOCITY)
    #print("min, max " , a, b)
    #ip.executeSampleData('Temperature')
    theNet().save("/work/sk_te/testgui.net")

def testData():
    ip = testImportModule1()
    #print(str(ip))
    value = 'Temperature'

    (a,b) = ip.getDataMinMax(value)
    if not (a,b)==('78.848465','446.557648'):
        print("(%s,%s) contra (78.848465,446.557648)" % (a,b))
        print("test failed")
    else:
        print("OK")

def testGroupGeo():

    gip = testLargeGroup()
    bp = gip.getBox()
    (x,y,z) = bp.getCenter()
    theNet().save("/work/sk_te/testgui.net")
    if not (x,y,z)==(1.6700575000000002, 0.0, 0.40254699999999999):
        print("test failed", (x,y,z))
    else:
        print("OK")


def testGroupData(gip):
    value = 'Temperature'
    c = Colors()
    theNet().add(c)
    connect( gip.dataConnectionPoint(value), ConnectionPoint( c, 'Data' ) )
    gip.executeData(value)

    coll = Collect()
    theNet().add(coll)
    connect( gip.geoConnectionPoint(value), ConnectionPoint( coll, 'grid' ) )
    theNet().connect( c, 'texture', coll, 'textures' )
    r = Renderer()
    theNet().add(r)
    theNet().connect( coll, 'geometry', r, 'RenderData')
    c.execute()

if __name__ == "__main__":
    #testGeo()
    #testData()
    #testSample()
    #testGroupGeo()
    testGroupData( testLargeSimpleGroup() )
