//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// MODULE        coGRMsg.h.i
//
// Description: SWIG interface definition
//
// Initial version: 10.03.2003 (rm@visenso.de)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by VirCinity IT Consulting
// (C) 2005 by Visual Engieering Solutions GmbH, Stuttgart
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
%module coGRMsg
%{
#include "grmsg/coGRMsg.h"
#include "grmsg/coGRObjMsg.h"
#include "grmsg/coGRObjRegisterMsg.h"
#include "grmsg/coGRObjVisMsg.h"
#include "grmsg/coGRObjMoveInterMsg.h"
#include "grmsg/coGRCreateViewpointMsg.h"
#include "grmsg/coGRCreateDefaultViewpointMsg.h"
#include "grmsg/coGRShowViewpointMsg.h"
#include "grmsg/coGRActivatedViewpointMsg.h"
#include "grmsg/coGRShowPresentationpointMsg.h"
#include "grmsg/coGRDeleteViewpointMsg.h"
#include "grmsg/coGRChangeViewpointMsg.h"
#include "grmsg/coGRChangeViewpointIdMsg.h"
#include "grmsg/coGRChangeViewpointNameMsg.h"
#include "grmsg/coGRViewpointChangedMsg.h"
#include "grmsg/coGRToggleFlymodeMsg.h"
#include "grmsg/coGRObjSetCaseMsg.h"
#include "grmsg/coGRObjSetNameMsg.h"
#include "grmsg/coGRObjMoveObjMsg.h"
#include "grmsg/coGRDocMsg.h"
#include "grmsg/coGRDocVisibleMsg.h"
#include "grmsg/coGRSetDocPageSizeMsg.h"
#include "grmsg/coGRSetDocPageMsg.h"
#include "grmsg/coGRSetDocScaleMsg.h"
#include "grmsg/coGRSetDocPositionMsg.h"
#include "grmsg/coGRAddDocMsg.h"
#include "grmsg/coGRObjBoundariesObjMsg.h"
#include "grmsg/coGRObjColorObjMsg.h"
#include "grmsg/coGRObjShaderObjMsg.h"
#include "grmsg/coGRObjMaterialObjMsg.h"
#include "grmsg/coGRObjSetTransparencyMsg.h"
#include "grmsg/coGRKeyWordMsg.h"
#include "grmsg/coGRObjTransformMsg.h"
#include "grmsg/coGRObjTransformCaseMsg.h"
#include "grmsg/coGRObjTransformSGItemMsg.h"
#include "grmsg/coGRObjRestrictAxisMsg.h"
#include "grmsg/coGRGraphicRessourceMsg.h"
#include "grmsg/coGRObjSetMoveMsg.h"
#include "grmsg/coGRObjSetMoveSelectedMsg.h"
#include "grmsg/coGRObjSensorMsg.h"
#include "grmsg/coGRObjSensorEventMsg.h"
#include "grmsg/coGRSendDocNumbersMsg.h"
#include "grmsg/coGRSendCurrentDocMsg.h"
#include "grmsg/coGRAnimationOnMsg.h"
#include "grmsg/coGRSetAnimationSpeedMsg.h"
#include "grmsg/coGRSetTimestepMsg.h"
#include "grmsg/coGRObjAttachedClipPlaneMsg.h"
#include "grmsg/coGRToggleVPClipPlaneModeMsg.h"
#include "grmsg/coGRSnapshotMsg.h"
#include "grmsg/coGRSetTrackingParamsMsg.h"
#include "grmsg/coGRObjMovedMsg.h"
#include "grmsg/coGRObjSetConnectionMsg.h"
#include "grmsg/coGRGenericParamRegisterMsg.h"
#include "grmsg/coGRGenericParamChangedMsg.h"
#include "grmsg/coGRObjSelectMsg.h"
#include "grmsg/coGRObjDelMsg.h"
#include "grmsg/coGRObjGeometryMsg.h"
#include "grmsg/coGRObjAddChildMsg.h"
#include "grmsg/coGRTurnTableAnimationMsg.h"
#include "grmsg/coGRObjSetVariantMsg.h"
#include "grmsg/coGRObjSetAppearanceMsg.h"
#include "grmsg/coGRObjKinematicsStateMsg.h"
%}
%include coGRMsg.h
%include coGRObjMsg.h
%include coGRObjRegisterMsg.h
%include coGRObjVisMsg.h
%include coGRObjMoveInterMsg.h
%include coGRObjMoveObjMsg.h
%include coGRCreateViewpointMsg.h
%include coGRCreateDefaultViewpointMsg.h
%include coGRShowViewpointMsg.h
%include coGRActivatedViewpointMsg.h
%include coGRDeleteViewpointMsg.h
%include coGRChangeViewpointMsg.h
%include coGRChangeViewpointIdMsg.h
%include coGRChangeViewpointNameMsg.h
%include coGRViewpointChangedMsg.h
%include coGRShowPresentationpointMsg.h
%include coGRObjSetCaseMsg.h
%include coGRObjSetNameMsg.h
%include coGRObjMoveObjMsg.h
%include coGRDocMsg.h
%include coGRDocVisibleMsg.h
%include coGRSetDocPositionMsg.h
%include coGRSetDocPageMsg.h
%include coGRSetDocPageSizeMsg.h
%include coGRSetDocScaleMsg.h
%include coGRAddDocMsg.h
%include coGRObjBoundariesObjMsg.h
%include coGRObjColorObjMsg.h
%include coGRObjShaderObjMsg.h
%include coGRObjMaterialObjMsg.h
%include coGRObjSetTransparencyMsg.h
%include coGRKeyWordMsg.h
%include coGRObjTransformAbstractMsg.h
%include coGRObjTransformMsg.h
%include coGRObjTransformCaseMsg.h
%include coGRObjTransformSGItemMsg.h
%include coGRToggleFlymodeMsg.h
%include coGRObjRestrictAxisMsg.h
%include coGRGraphicRessourceMsg.h
%include coGRObjSetMoveMsg.h
%include coGRObjSetMoveSelectedMsg.h
%include coGRObjSensorMsg.h
%include coGRObjSensorEventMsg.h
%include coGRSendDocNumbersMsg.h
%include coGRSendCurrentDocMsg.h
%include coGRAnimationOnMsg.h
%include coGRSetAnimationSpeedMsg.h
%include coGRSetTimestepMsg.h
%include coGRObjAttachedClipPlaneMsg.h
%include coGRToggleVPClipPlaneModeMsg.h
%include coGRSnapshotMsg.h
%include coGRSetTrackingParamsMsg.h
%include coGRObjMovedMsg.h
%include coGRObjSetConnectionMsg.h
%include coGRGenericParamRegisterMsg.h
%include coGRGenericParamChangedMsg.h
%include coGRObjSelectMsg.h
%include coGRObjDelMsg.h
%include coGRObjGeometryMsg.h
%include coGRObjAddChildMsg.h
%include coGRTurnTableAnimationMsg.h
%include coGRObjSetVariantMsg.h
%include coGRObjSetAppearanceMsg.h
%include coGRObjKinematicsStateMsg.h


