/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLUGINUTILS_PLUGINMESSAGETYPE_H
#define PLUGINUTILS_PLUGINMESSAGETYPE_H

#include <OpenVRUI/sginterface/vruiCollabInterface.h>
namespace opencover
{
class PluginMessageTypes
{

public:
    enum Type
    {
        VruiCollabInterfaceNone = vrui::vruiCollabInterface::NONE,                     // = 10
        VruiCollabInterfaceValuePoti = vrui::vruiCollabInterface::VALUEPOTI,           // = 11
        VruiCollabInterfaceHSVWheel = vrui::vruiCollabInterface::HSVWHEEL,             // = 12
        VruiCollabInterfacePushButton = vrui::vruiCollabInterface::PUSHBUTTON,         // = 13
        VruiCollabInterfaceToggleButton = vrui::vruiCollabInterface::TOGGLEBUTTON,     // = 14
        VruiCollabInterfaceFunctionEditor = vrui::vruiCollabInterface::FunctionEditor, // = 15
        VruiCollabInterfacePinEditor = vrui::vruiCollabInterface::PinEditor,           // = 16
        AKToolbarInactive = 20,
        AKToolbarActive = 21,
        AnnotationMessage = 30,
        AnnotationTextMessage = 31,
        ClipPlaneMessage = 35,
        PLMXMLLoadFiles = 55,
        PLMXMLHideNode = 56,
        PLMXMLShowNode = 57,
        PLMXMLSetSimPair = 58,
        Marker0 = 60,
        Marker1,
        Bullet0 = 60,
        Bullet1,
        MeasureKit0 = 70,
        MeasureKit1,
        MoveAddMoveNode = 80,
        MoveMoveNode,
        MoveMoveNodeFinished,
        Measure0 = 90,
        Measure1,
        PBufferDoSnap = 100,
        PBufferDoSnapFile = 101,
        PBufferDoneSnapshot = 102,
        PDBLoadFile = 150,
        PDBLoadAni = 151,
        PDBHighDetail = 154,
        PDBMoveMark = 164,
        PDBMisc = 166,
        VolumeLoadFile = 250,
        VolumeROIMsg = 251,
        VolumeClipMsg = 252,
        NurbsSurfacePointMsg = 260,
        PointCloudSurfaceMsg = 261,
        PointCloudSelectionIsBoundaryMsg = 262,
        PointCloudSelectionSetMsg = 263,
        Browser = 264,
        VideoStartCapture = 265,
        VideoEndCapture = 266,

        // COVERScript
        COVERScriptEvaluate = 300,
        COVERScriptLoad = 301,
        RemoteDTConnectToHost = 1230,
        RemoteDTDisconnect,
        RemoteDTShowDesktop,
        RemoteDTHideDesktop,
        WSInterfaceCustomMessage = 1240,

        // Variant Messages
        VariantHide = 1301,
        VariantShow = 1302,
        SGBrowserHideNode = 2000,
        SGBrowserShowNode = 2001,
        SGBrowserSetProperties = 2002,
        SGBrowserGetProperties = 2003,
        SGBrowserRemoveTexture = 2004,
        SGBrowserSetTexture = 2005,
        SGBrowserRemoveShader = 2006,
        SGBrowserSetShader = 2007,
        SGBrowserSetUniform = 2008,
        SGBrowserSetVertex = 2009,
        SGBrowserSetFragment = 2010,
        SGBrowserSetGeometry = 2011,
        SGBrowserSetTessControl = 2012,
        SGBrowserSetTessEval = 2013,
        SGBrowserSetNumVertex = 2014,
        SGBrowserSetInputType = 2015,
        SGBrowserSetOutputType = 2016,

        // Vistle messages
        VistleMessageIn = 9000,  // distribute message received from Vistle to plugins
        VistleMessageOut = 9001, // let Vistle plugin forward message to Vistle

        // HLRS plugins start with 10000 ----------------------------------------
        HLRS_ACInterfaceSnapshotPath = 10333,
        HLRS_ACInterfaceModelLoadedPath,
        HLRS_SteeringWheelRemoteVehiclePosition = 10367,
        HLRS_cuCuttingSurface = 10400,
        HLRS_Revit_Message = 10500,
        HLRS_Office_Message = 10600,
        HLRS_Oddlot_Message = 10700,

        // Calit2 plugins start with 20000 --------------------------------------
        Calit2_FileBrowserRegisterExt = 20001,
        Calit2_FileBrowserReleaseExt = 20002,
        Calit2_CCDBVRLoadFile = 20150,
        Calit2_CCDBVRMoveStructure = 20151,
        Calit2_CCDBVRResetStructure = 20152,
        Calit2_CCDBVRClearAll = 20153,
        Calit2_CCDBVRColorStructure = 20154,
        Calit2_CCDBVRAlphaStructure = 20155,
        Calit2_CCDBVRLabelStructure = 20156,
        Calit2_CCDBVRClearLabels = 20157,
        Calit2_CCDBVRModelCounter = 20158,
        Calit2_CCDBVRNestModels = 20159,
        Calit2_CCDBVRAnnotationPartOf = 20160,
        Calit2_CCDBVRAnnotationContinuous = 20161,
        Calit2_CCDBVRRedrawTree = 20162,
        Calit2_CCDBVRAnnotationAddRoot = 20163,
        Calit2_CCDBVRIncNumLoaded = 20164,
        Calit2_Image = 20200,

        // RRZK -----------------------------------------------------------------
        RRZK_rrxevent = 30000,
    };
};
}

#endif
