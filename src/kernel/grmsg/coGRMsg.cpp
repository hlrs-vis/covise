/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>

#include "coGRActivatedViewpointMsg.h"
#include "coGRAddDocMsg.h"
#include "coGRAnimationOnMsg.h"
#include "coGRChangeViewpointIdMsg.h"
#include "coGRChangeViewpointMsg.h"
#include "coGRChangeViewpointNameMsg.h"
#include "coGRCreateDefaultViewpointMsg.h"
#include "coGRCreateViewpointMsg.h"
#include "coGRDeleteViewpointMsg.h"
#include "coGRDocMsg.h"
#include "coGRDocVisibleMsg.h"
#include "coGRGenericParamChangedMsg.h"
#include "coGRGenericParamRegisterMsg.h"
#include "coGRGraphicRessourceMsg.h"
#include "coGRKeyWordMsg.h"
#include "coGRObjAddChildMsg.h"
#include "coGRObjAttachedClipPlaneMsg.h"
#include "coGRObjBoundariesObjMsg.h"
#include "coGRObjColorObjMsg.h"
#include "coGRObjDelMsg.h"
#include "coGRObjGeometryMsg.h"
#include "coGRObjKinematicsStateMsg.h"
#include "coGRObjMaterialObjMsg.h"
#include "coGRObjMoveInterMsg.h"
#include "coGRObjMoveObjMsg.h"
#include "coGRObjMovedMsg.h"
#include "coGRObjMsg.h"
#include "coGRObjRegisterMsg.h"
#include "coGRObjRestrictAxisMsg.h"
#include "coGRObjSelectMsg.h"
#include "coGRObjSensorEventMsg.h"
#include "coGRObjSensorMsg.h"
#include "coGRObjSetAppearanceMsg.h"
#include "coGRObjSetCaseMsg.h"
#include "coGRObjSetConnectionMsg.h"
#include "coGRObjSetMoveMsg.h"
#include "coGRObjSetMoveSelectedMsg.h"
#include "coGRObjSetNameMsg.h"
#include "coGRObjSetTransparencyMsg.h"
#include "coGRObjSetVariantMsg.h"
#include "coGRObjShaderObjMsg.h"
#include "coGRObjTransformAbstractMsg.h"
#include "coGRObjTransformCaseMsg.h"
#include "coGRObjTransformMsg.h"
#include "coGRObjTransformSGItemMsg.h"
#include "coGRObjVisMsg.h"
#include "coGRPluginMsg.h"
#include "coGRSendCurrentDocMsg.h"
#include "coGRSendDocNumbersMsg.h"
#include "coGRSetAnimationSpeedMsg.h"
#include "coGRSetDocPageMsg.h"
#include "coGRSetDocPageSizeMsg.h"
#include "coGRSetDocPositionMsg.h"
#include "coGRSetDocScaleMsg.h"
#include "coGRSetTimestepMsg.h"
#include "coGRSetTrackingParamsMsg.h"
#include "coGRSetViewpointFile.h"
#include "coGRShowPresentationpointMsg.h"
#include "coGRShowViewpointMsg.h"
#include "coGRSnapshotMsg.h"
#include "coGRToggleFlymodeMsg.h"
#include "coGRToggleVPClipPlaneModeMsg.h"
#include "coGRTurnTableAnimationMsg.h"
#include "coGRViewpointChangedMsg.h"



using namespace std;
using namespace grmsg;

#define CREATE_GRMSG(enumName, className)                                 \
    case coGRMsg::Mtype::enumName:                                                 \
    {                                                                     \
        return std::unique_ptr<className>(new className(fullmsg.c_str())); \
    }

/// split message string by this token
static const char SplitToken = '\n';

/// string put to the header to identify the type Gui2RenderMessage
static const std::string MsgID = "GRMSG";

std::unique_ptr<coGRMsg> grmsg::create(const char *msg)
{
   if(msg[0] == '\0' && MsgID == msg + 1)
   {
       const char *pos = msg + MsgID.size() + 2;
       size_t num = 0;
       while (*pos != SplitToken)
       {
           ++pos;
           ++num;
       }
       string t{msg + MsgID.size() + 2, num};
       coGRMsg::Mtype type = (coGRMsg::Mtype)std::stoi(t);
       string fullmsg = MsgID + SplitToken + (msg + MsgID.size() + 2);
       
       switch (type)
       {
        CREATE_GRMSG(REGISTER, coGRObjRegisterMsg)
        CREATE_GRMSG(GEO_VISIBLE, coGRObjVisMsg)
        CREATE_GRMSG(INTERACTOR_VISIBLE, coGRObjVisMsg)
        CREATE_GRMSG(SMOKE_VISIBLE, coGRObjVisMsg)
        CREATE_GRMSG(MOVE_INTERACTOR, coGRObjMoveInterMsg)
        CREATE_GRMSG(INTERACTOR_USED, coGRObjVisMsg)
        CREATE_GRMSG(CREATE_VIEWPOINT, coGRCreateViewpointMsg)
        CREATE_GRMSG(CREATE_DEFAULT_VIEWPOINT, coGRCreateDefaultViewpointMsg)
        CREATE_GRMSG(SHOW_VIEWPOINT, coGRShowViewpointMsg)
        CREATE_GRMSG(SHOW_PRESENTATIONPOINT, coGRShowPresentationpointMsg)
        CREATE_GRMSG(CHANGE_VIEWPOINT_ID, coGRChangeViewpointIdMsg)
        CREATE_GRMSG(CHANGE_VIEWPOINT_NAME, coGRChangeViewpointNameMsg)
        CREATE_GRMSG(DELETE_VIEWPOINT, coGRDeleteViewpointMsg)
        CREATE_GRMSG(SET_VIEWPOINT_FILE, coGRSetViewpointFile)
        CREATE_GRMSG(FLYMODE_TOGGLE, coGRToggleFlymodeMsg)
        CREATE_GRMSG(SET_CASE, coGRObjSetCaseMsg)
        CREATE_GRMSG(SET_NAME, coGRObjSetNameMsg)
        CREATE_GRMSG(MOVE_OBJECT, coGRObjMoveObjMsg)
        CREATE_GRMSG(ADD_DOCUMENT, coGRAddDocMsg)
        CREATE_GRMSG(SET_DOCUMENT_PAGE, coGRSetDocPageMsg)
        CREATE_GRMSG(SET_DOCUMENT_SCALE, coGRSetDocScaleMsg)
        CREATE_GRMSG(SET_DOCUMENT_POSITION, coGRSetDocPositionMsg)
        CREATE_GRMSG(SET_DOCUMENT_PAGESIZE, coGRSetDocPageSizeMsg)
        CREATE_GRMSG(SEND_DOCUMENT_NUMBERS, coGRSendDocNumbersMsg)
        CREATE_GRMSG(DOC_VISIBLE, coGRDocVisibleMsg)
        CREATE_GRMSG(BOUNDARIES_OBJECT, coGRObjBoundariesObjMsg)
        CREATE_GRMSG(COLOR_OBJECT, coGRObjColorObjMsg)
        CREATE_GRMSG(SHADER_OBJECT, coGRObjShaderObjMsg)
        CREATE_GRMSG(MATERIAL_OBJECT, coGRObjMaterialObjMsg)
        CREATE_GRMSG(SET_TRANSPARENCY, coGRObjSetTransparencyMsg)
        CREATE_GRMSG(KEYWORD,coGRKeyWordMsg )
        CREATE_GRMSG(TRANSFORM_OBJECT, coGRObjTransformMsg)
        CREATE_GRMSG(TRANSFORM_CASE, coGRObjTransformCaseMsg)
        CREATE_GRMSG(RESTRICT_AXIS, coGRObjRestrictAxisMsg)
        CREATE_GRMSG(GRAPHIC_RESSOURCE, coGRGraphicRessourceMsg)
        CREATE_GRMSG(SET_MOVE, coGRObjSetMoveMsg)
        CREATE_GRMSG(SET_MOVE_SELECTED, coGRObjSetMoveSelectedMsg)
        CREATE_GRMSG(SENSOR, coGRObjSensorMsg)
        CREATE_GRMSG(SENSOR_EVENT, coGRObjSensorEventMsg)
        CREATE_GRMSG(ATTACHED_CLIPPLANE, coGRObjAttachedClipPlaneMsg)
        CREATE_GRMSG(ANIMATION_ON, coGRAnimationOnMsg)
        CREATE_GRMSG(ANIMATION_SPEED, coGRSetAnimationSpeedMsg)
        CREATE_GRMSG(ANIMATION_TIMESTEP, coGRSetTimestepMsg)
        CREATE_GRMSG(ACTIVATED_VIEWPOINT, coGRActivatedViewpointMsg)
        CREATE_GRMSG(VPCLIPPLANEMODE_TOGGLE, coGRToggleVPClipPlaneModeMsg)
        CREATE_GRMSG(SNAPSHOT, coGRSnapshotMsg)
        CREATE_GRMSG(SEND_CURRENT_DOCUMENT, coGRSendCurrentDocMsg)
        CREATE_GRMSG(SET_TRACKING_PARAMS, coGRSetTrackingParamsMsg)
        CREATE_GRMSG(CHANGE_VIEWPOINT, coGRChangeViewpointMsg)
        CREATE_GRMSG(VIEWPOINT_CHANGED, coGRViewpointChangedMsg)
        CREATE_GRMSG(OBJECT_TRANSFORMED, coGRObjMovedMsg)
        CREATE_GRMSG(SET_CONNECTIONPOINT, coGRObjSetConnectionMsg)
        CREATE_GRMSG(GENERIC_PARAM_REGISTER, coGRGenericParamRegisterMsg)
        CREATE_GRMSG(GENERIC_PARAM_CHANGED, coGRGenericParamChangedMsg)
        CREATE_GRMSG(TRANSFORM_SGITEM, coGRObjTransformSGItemMsg)
        CREATE_GRMSG(SELECT_OBJECT, coGRObjSelectMsg)
        CREATE_GRMSG(DELETE_OBJECT, coGRObjDelMsg)
        CREATE_GRMSG(GEOMETRY_OBJECT, coGRObjGeometryMsg)
        CREATE_GRMSG(ADD_CHILD_OBJECT, coGRObjAddChildMsg)
        CREATE_GRMSG(TURNTABLE_ANIMATION, coGRTurnTableAnimationMsg)
        CREATE_GRMSG(SET_VARIANT, coGRObjSetVariantMsg)
        CREATE_GRMSG(SET_APPEARANCE, coGRObjSetAppearanceMsg)
        CREATE_GRMSG(KINEMATICS_STATE, coGRObjKinematicsStateMsg)
        CREATE_GRMSG(PLUGIN, coGRPluginMsg)
        default:
            return nullptr;
         }
   }
   return nullptr;
}


coGRMsg::coGRMsg(Mtype type): is_valid_(type != NO_TYPE), type_(type)
{
    addToken(MsgID.c_str());
    addToken(std::to_string(type).c_str());
}

coGRMsg::coGRMsg(const char *msg)
    : is_valid_(true)
    ,content_(msg)
{
    /// parsed content will always be removed
    /// to enable all children to call extractFirstToken() to
    /// get their token

    /// parse for msg id
    if (extractFirstToken() != MsgID)
    {
        is_valid_ = false;
        return;
    }

    /// parse message type
    type_ = (Mtype)std::stoi(extractFirstToken());
    if (type_ == NO_TYPE)
    {
        is_valid_ = 0;
    }
}


void coGRMsg::addToken(const char *token)
{
    if (token)
        content_ += string(token) + SplitToken;
}

string coGRMsg::getFirstToken()
{
    size_t pos = content_.find(SplitToken);
    return content_.substr(0, pos);
}

string coGRMsg::extractFirstToken()
{
    size_t pos = content_.find(SplitToken);
    string token = content_.substr(0, pos);
    content_ = content_.substr(pos + 1, string::npos);
    return token;
}

vector<string> coGRMsg::getAllTokens()
{
    vector<string> tok;

    istringstream s(content_);
    string temp;

    while (std::getline(s, temp, SplitToken))
    {
        tok.push_back(temp);
    }
    return tok;
}

const char *coGRMsg::c_str() const
{
    return content_.c_str();
}

string coGRMsg::getString() const
{
    return content_;
}

void coGRMsg::print_stdout()
{
    const char *typeStr[] = { "NO_TYPE", "GEO_VISIBLE", "REGISTER", "INTERACTOR_VISIBLE" };
    cout << "coGRMsg::Type = " << typeStr[type_] << endl;
    cout << "coGRMsg::content = " << endl << "----" << endl << content_ << endl << "---- " << endl;
}
