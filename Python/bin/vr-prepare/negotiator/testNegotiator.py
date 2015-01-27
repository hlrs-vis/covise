
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

"""Example for the use of the entities provided by initHandlers().

Start with
$ covise --script <name of this module>

"""



from coviseCase import *
from KeydObject import globalKeyHandler, RUN_ALL, RUN_OCT

import pdb
import sys
import time

from qt import *

from negGuiHandlers import theGuiMsgHandler, initHandlers
from Neg2GuiMessages import SESSION_KEY
from KeydObject import ( TYPE_PROJECT, TYPE_CASE, TYPE_2D_GROUP, TYPE_3D_GROUP, TYPE_2D_PART, TYPE_3D_PART,
     VIS_2D_STATIC_COLOR, VIS_2D_SCALAR_COLOR, VIS_2D_RAW, VIS_3D_BOUNDING_BOX, VIS_STREAMLINE, RUN_OCT, VIS_VRML, VIS_COVISE,
     VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE, VIS_DOCUMENT, TYPE_CAD_PRODUCT, VIS_STREAMLINE_2D, VIS_DOMAINLINES)

from coCaseMgr import coCaseMgrParams
from Part2DStaticColorVis import Part2DStaticColorVisParams
from Part2DScalarColorVis import Part2DScalarColorVisParams
from PartTracerVis import PartStreamlineVisParams,PartStreamline2DVisParams 
from VisItem import VisItemParams
from coColorTable import coColorTable, coColorTableParams
from coColorCreator import coColorCreator, coColorCreatorParams
from VRPCoviseNetAccess import theNet
from coviseCase import (coviseCase2DimensionSeperatedCase, NameAndCoviseCase )
from VrmlVis import VrmlVisParams
from CoviseVis import CoviseVisParams
from PartCuttingSurfaceVis import PartPlaneVisParams, PartVectorVisParams
from PartIsoSurfaceVis import PartIsoSurfaceVisParams
from coDocumentMgr import coDocumentMgrParams, coDocumentMgr
from coCADMgr import coCADMgrParams
from Utils import Line3D

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                            """
"""                         T E S T S                          """
"""                                                            """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


def createProjectPanel(requestNr, typeNr, key, parentKey, params):
    print("create project panel with key %s of parent %s " % (key,parentKey))
    global project_key
    project_key = key
    theGuiMsgHandler().registerParamCallback(key, setProjectPanelParams)
    theGuiMsgHandler().registerAddCallback(key, createCasePanel)
    theGuiMsgHandler().answerOk(requestNr)

def setProjectPanelParams(requestNr, key, params):
    print("setProjectPanelParams; date is now : %s " % params.creation_date)
    theGuiMsgHandler().answerOk(requestNr)


def createCasePanel(requestNr, typeNr, key, parentKey, params):
    print("create Case panel with key %s of parent %s " % (key,parentKey))
    global case_key1
    case_key1 = key
    theGuiMsgHandler().registerParamCallback(key, setCasePanelParams)
    theGuiMsgHandler().registerAddCallback(key, createGroupPanel)
    theGuiMsgHandler().answerOk(requestNr)

def setCasePanelParams(requestNr, key, params):
    print("setCasePanelParams; name is now : %s " % params.name)
    theGuiMsgHandler().answerOk(requestNr)

Group_key1=-1
def createGroupPanel(requestNr, typeNr, key, parentKey, params):
    print("create Group panel with key %s of parent %s " % (key,parentKey))
    global Group_key1
    global Group_key2
    Group_key1 = key

    theGuiMsgHandler().registerParamCallback(key, setGroupPanelParams)
    theGuiMsgHandler().registerAddCallback(key, createPartPanel)
    theGuiMsgHandler().answerOk(requestNr)


Part_Key = -1
def createPartPanel( requestNr, typeNr, key, parentKey, params):
    print("create Part panel with key %s of parent %s " % (key,parentKey))
    global Part_Key
    global Part_Key_3D
    if Part_Key==-1:
        Part_Key = key
        if not ( testMode==VIS_3D_BOUNDING_BOX or testMode==VIS_STREAMLINE or testMode==VIS_PLANE or testMode==VIS_VECTOR or testMode==VIS_ISOPLANE):
            if testMode==VIS_STREAMLINE_2D:
                theGuiMsgHandler().registerAddCallback(key, createTracePanel)
            else:
                theGuiMsgHandler().registerAddCallback(key, createStaticColorVis)
            theGuiMsgHandler().registerParamCallback(key, setPartPanelParams)
            theGuiMsgHandler().answerOk(requestNr)
            # test object testMode
            reqId = theGuiMsgHandler().requestObject(
                testMode, None, Part_Key )
            theGuiMsgHandler().waitforAnswer(reqId)
    elif typeNr==TYPE_3D_PART :
        Part_Key_3D = key
        if not testMode==VIS_STREAMLINE_2D:
            if testMode==VIS_3D_BOUNDING_BOX or testMode==VIS_DOMAINLINES or testMode==VIS_STREAMLINE or testMode==VIS_PLANE or testMode==VIS_VECTOR or testMode==VIS_ISOPLANE:
                theGuiMsgHandler().registerParamCallback(key, setPartPanelParams)
                if testMode==VIS_STREAMLINE or testMode==VIS_PLANE or testMode==VIS_VECTOR or testMode==VIS_ISOPLANE:
                    theGuiMsgHandler().registerAddCallback(key, createTracePanel)
                reqId = theGuiMsgHandler().requestObject(
                    testMode, None, key )
                theGuiMsgHandler().waitforAnswer(reqId)
    theGuiMsgHandler().answerOk(requestNr)

Tracer_key = -1
def createTracePanel( requestNr, typeNr, key, parentKey, params):
    print("create trace panel with key %s of parent %s " % (key,parentKey))
    global Tracer_key
    if Tracer_key==-1:
        Tracer_key = key
    theGuiMsgHandler().answerOk(requestNr)

VisColor_key = -1
def createStaticColorVis( requestNr, typeNr, key, parentKey, params):
    print("create static color panel with key %s of parent %s " % (key,parentKey))
    global VisColor_key
    if VisColor_key==-1:
        VisColor_key = key
    theGuiMsgHandler().answerOk(requestNr)

def setGroupPanelParams(requestNr, key, params):
    print("setGroupPanelParams; file is now : %s " % params.name)
    theGuiMsgHandler().answerOk(requestNr)

def setPartPanelParams(requestNr, key, params):
    print("setPartPanelParams : ")
    global scalar_variable
    global vector_variable
    scalar_variable = None
    for v in params.partcase.variables:
        print("Variable %s in file %s " % ( v.name, v.filename ))
        if scalar_variable==None:
            scalar_variable=v.name
        if v.variableDimension==3:
            print("Choose Velocity ", v.filename)
            vector_variable=v.name
    theGuiMsgHandler().answerOk(requestNr)

def status():
    print("Status")
    print("------")
    print(str(globalKeyHandler().getObject(project_key)))

def testRequest():
    """Example for requesting something at the theGuiMsgHandler()."""
    keepReference = QApplication(sys.argv)

    initHandlers()

    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, createProjectPanel)
    reqId = g.requestObject(TYPE_PROJECT)
    g.waitforAnswer(reqId)

    reqId = g.requestObject(TYPE_CASE, None, project_key)
    g.waitforAnswer(reqId)

    caseP = coCaseMgrParams()
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/'
                            'TINY/CoviseDaten/TINY.cocase')
    #nameAndCase.setFromFile(
    #        '/work/common/Projekte/DC-CFDGui/datasets/'
    #        'msport/CoviseDaten/msport.cocase')

    caseP.origDsc = caseP.filteredDsc = coviseCase2DimensionSeperatedCase(nameAndCase.case, nameAndCase.name, nameAndCase.pathToCaseFile)
    g.setParams(case_key1, caseP)

    # testing static coloring
    if testMode==VIS_2D_STATIC_COLOR:
        colP = Part2DStaticColorVisParams()
        colP.g = 0
        g.setParams( VisColor_key, colP )

        status()

        reqId = g.runObject( case_key1 )
        g.waitforAnswer(reqId)

        colP.r = 255
        colP.g = 0
        colP.b = 0
        g.setParams( VisColor_key, colP )
        g.runObject( case_key1 )


    #testing scalar coloring
    elif testMode==VIS_2D_SCALAR_COLOR:
        colP = Part2DScalarColorVisParams()
        colP.variable = scalar_variable
        g.setParams( VisColor_key, colP )
        g.runObject( case_key1 )

    elif testMode==VIS_STREAMLINE:
        streamP = PartStreamlineVisParams()
        streamP.variable = vector_variable
        """
        colorP = coColorCreator()
        ctableP = coColorTableParams()
        ctableP.min = 98
        ctableP.max = 100
        ctableP.colorMapIdx = 2

        ctable = coColorTable()
        ctable.params = ctableP
        colorP.params.colorTable = ctable
        streamP.colorCreator = colorP
        colorP.run(RUN_ALL)
        """
        g.setParams( Tracer_key, streamP )
        r = g.runObject( case_key1, RUN_OCT )
        g.waitforAnswer(r)
        g.runObject( Tracer_key )

        #g.saveObject(0, "/work/sk_te/test.coprj")
        theNet().save("/work/ko_te/testgui.net")
    
    elif testMode==VIS_STREAMLINE_2D:
        streamP = PartStreamline2DVisParams()
        streamP.variable = vector_variable
        streamP.alignedRectangle = Line3D()
        streamP.alignedRectangle.setStartEndPoint(0.1, -0.6, 0., 0.5, -0.8, 0. )
        
        g.setParams( Tracer_key, streamP )
        r = g.runObject( case_key1, RUN_OCT )
        g.waitforAnswer(r)
        g.runObject( Tracer_key )
        theNet().save("/work/sk_te/test.net")
        status()
        
    elif testMode==VIS_PLANE:
        planeP = PartPlaneVisParams()
        planeP.variable = vector_variable
        planeP.alignedRectangle.middle = (0, 0, 1)
        """
        colorP = coColorCreator()
        ctableP = coColorTableParams()
        ctableP.min = 98
        ctableP.max = 100
        ctableP.colorMapIdx = 2

        ctable = coColorTable()
        ctable.params = ctableP
        colorP.params.colorTable = ctable
        streamP.colorCreator = colorP
        colorP.run(RUN_ALL)
        """
        g.setParams( Tracer_key, planeP )
        r = g.runObject( case_key1, RUN_OCT )
        g.waitforAnswer(r)
        g.runObject( Tracer_key )

        #g.saveObject(0, "/work/ko_te/test.coprj")
        theNet().save("/work/ko_te/testgui.net")

    elif testMode==VIS_VECTOR:
        vectorP = PartVectorVisParams()
        vectorP.variable = vector_variable
        """
        colorP = coColorCreator()
        ctableP = coColorTableParams()
        ctableP.min = 98
        ctableP.max = 100
        ctableP.colorMapIdx = 2

        ctable = coColorTable()
        ctable.params = ctableP
        colorP.params.colorTable = ctable
        streamP.colorCreator = colorP
        colorP.run(RUN_ALL)
        """
        g.setParams( Tracer_key, vectorP )
        r = g.runObject( case_key1, RUN_OCT )
        g.waitforAnswer(r)
        g.runObject( Tracer_key )

        #g.saveObject(0, "/work/ko_te/test.coprj")
        theNet().save("/work/ko_te/testgui.net")

    elif testMode==VIS_ISOPLANE:
        isoplaneP = PartIsoSurfaceVisParams()
        isoplaneP.variable = vector_variable
        """
        colorP = coColorCreator()
        ctableP = coColorTableParams()
        ctableP.min = 98
        ctableP.max = 100
        ctableP.colorMapIdx = 2

        ctable = coColorTable()
        ctable.params = ctableP
        colorP.params.colorTable = ctable
        streamP.colorCreator = colorP
        colorP.run(RUN_ALL)
        """
        g.setParams( Tracer_key, isoplaneP )
        r = g.runObject( case_key1, RUN_OCT )
        g.waitforAnswer(r)
        g.runObject( Tracer_key )

        #g.saveObject(0, "/work/ko_te/test.coprj")
        theNet().save("/work/ko_te/testgui.net")

    else :
        status()
        g.runObject( case_key1 )
        vP = VisItemParams()
        vP.isVisible=True
        g.setParams( VisColor_key, vP )
        theNet().save("/work/sk_te/testgui.net")

    if False: # True:
        reqId = g.requestObject(TYPE_CASE, None, project_key)
        g.waitforAnswer(reqId)
        caseP = coCaseMgrParams()
        caseP.filename = (
            '/work/common/Projekte/DC-CFDGui/datasets/'
            'msport/CoviseDaten/msport.cocase')
        g.setParams(case_key1, caseP)

def testVRML():
    """test reading vrml"""
    keepReference = QApplication(sys.argv)

    initHandlers()

    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, createProjectPanel)
    reqId = g.requestObject(TYPE_PROJECT)
    g.waitforAnswer(reqId)

    reqId = g.requestObject(VIS_VRML, None, project_key)
    g.waitforAnswer(reqId)

    vrmlP = VrmlVisParams()
    vrmlP.modelPath='/work/ko_te/test.wrl'
    g.setParams(case_key1, vrmlP)
    g.runObject(case_key1)

    theNet().save("/work/ko_te/testVrml.net")

def testCovise():
    """test reading covise"""
    keepReference = QApplication(sys.argv)

    initHandlers()

    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, createProjectPanel)
    reqId = g.requestObject(TYPE_PROJECT)
    g.waitforAnswer(reqId)

    reqId = g.requestObject(VIS_COVISE, None, project_key)
    g.waitforAnswer(reqId)

    coviseP = CoviseVisParams()
    coviseP.grid_path='/data/general/examples/Covise/airbag.covise'
    g.setParams(case_key1, coviseP)
    g.runObject(case_key1)

    theNet().save("/work/ko_te/testCovise.net")

def testDocument():
    """test reading covise"""
    keepReference = QApplication(sys.argv)

    initHandlers()

    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, createProjectPanel)
    reqId = g.requestObject(TYPE_PROJECT)
    g.waitforAnswer(reqId)

    time.sleep(2)

    reqId = g.requestObject(VIS_DOCUMENT, None, project_key)
    g.waitforAnswer(reqId)

    coviseP = coDocumentMgrParams()
    coviseP.pos = (100., 0., 100.)
    coviseP.isVisible = True
    coviseP.documentName = 'Mein Auge'
    coviseP.imageName    = '/data/general/examples/Images/auge_blau.tif'
    g.setParams(4, coviseP)
    #g.runObject(case_key1)

def testCAD():
    """test reading covise"""
    keepReference = QApplication(sys.argv)
    initHandlers()
    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, createProjectPanel)
    reqId = g.requestObject(TYPE_PROJECT)
    g.waitforAnswer(reqId)

    time.sleep(2)

    reqId = g.requestObject(TYPE_CAD_PRODUCT, None, project_key)
    g.waitforAnswer(reqId)

    print("ICI")
    coviseP = coCADMgrParams()
    coviseP.filename = '/data/Kunden/Kaercher/catiav5/Wagen/Reifen.CATPart'
    reqId = g.setParams(4, coviseP)

    reqId = theGuiMsgHandler().requestObject( VIS_2D_RAW, None, 5 )
    g.waitforAnswer(reqId)

    g.runObject(4)

if __name__ == "__main__":
    # pdb.run('testRequest()')
    testMode = VIS_DOMAINLINES # VIS_STREAMLINE_2D VIS_3D_BOUNDING_BOX#VIS_ISOPLANE#VIS_VECTOR#VIS_PLANE# VIS_2D_RAW #VIS_STREAMLINE #VIS_3D_BOUNDING_BOX #VIS_2D_SCALAR_COLOR #VIS_2D_STATIC_COLOR
    testRequest()
#   testCAD()
#    testVRML()
#    testCovise()

# eof
