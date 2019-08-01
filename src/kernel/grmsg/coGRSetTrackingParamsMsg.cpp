/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetTrackingParamsMsg.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace std;
using namespace grmsg;

GRMSGEXPORT coGRSetTrackingParamsMsg::coGRSetTrackingParamsMsg(
    bool rotatePoint,
    bool rotatePointShown,
    float rotationPointSize,
    float rotatePointX,
    float rotatePointY,
    float rotatePointZ,
    bool rotateAxis,
    float rotateAxisX,
    float rotateAxisY,
    float rotateAxisZ,
    bool translateRestrict,
    float translateMinX,
    float translateMaxX,
    float translateMinY,
    float translateMaxY,
    float translateMinZ,
    float translateMaxZ,
    float translateFactor,
    bool scaleRestrict,
    float scaleMin,
    float scaleMax,
    float scaleFactor,
    bool enableTracking,
    const char *navigationMode)
    : coGRMsg(SET_TRACKING_PARAMS)
{
    rotatePoint_ = rotatePoint;
    rotatePointShown_ = rotatePointShown;
    rotationPointSize_ = rotationPointSize;
    rotatePointX_ = rotatePointX;
    rotatePointY_ = rotatePointY;
    rotatePointZ_ = rotatePointZ;
    rotateAxis_ = rotateAxis;
    rotateAxisX_ = rotateAxisX;
    rotateAxisY_ = rotateAxisY;
    rotateAxisZ_ = rotateAxisZ;
    translateRestrict_ = translateRestrict;
    translateMinX_ = translateMinX;
    translateMaxX_ = translateMaxX;
    translateMinY_ = translateMinY;
    translateMaxY_ = translateMaxY;
    translateMinZ_ = translateMinZ;
    translateMaxZ_ = translateMaxZ;
    translateFactor_ = translateFactor;
    scaleRestrict_ = scaleRestrict;
    scaleMin_ = scaleMin;
    scaleMax_ = scaleMax;
    scaleFactor_ = scaleFactor;
    trackingEnable_ = enableTracking;

    char str[1024];
    sprintf(str, "%d", rotatePoint);
    addToken(str);
    sprintf(str, "%d", rotatePointShown);
    addToken(str);
    sprintf(str, "%f", rotationPointSize);
    addToken(str);
    sprintf(str, "%f", rotatePointX);
    addToken(str);
    sprintf(str, "%f", rotatePointY);
    addToken(str);
    sprintf(str, "%f", rotatePointZ);
    addToken(str);
    sprintf(str, "%d", rotateAxis);
    addToken(str);
    sprintf(str, "%f", rotateAxisX);
    addToken(str);
    sprintf(str, "%f", rotateAxisY);
    addToken(str);
    sprintf(str, "%f", rotateAxisZ);
    addToken(str);
    sprintf(str, "%d", translateRestrict);
    addToken(str);
    sprintf(str, "%f", translateMinX);
    addToken(str);
    sprintf(str, "%f", translateMaxX);
    addToken(str);
    sprintf(str, "%f", translateMinY);
    addToken(str);
    sprintf(str, "%f", translateMaxY);
    addToken(str);
    sprintf(str, "%f", translateMinZ);
    addToken(str);
    sprintf(str, "%f", translateMaxZ);
    addToken(str);
    sprintf(str, "%f", translateFactor);
    addToken(str);
    sprintf(str, "%d", scaleRestrict);
    addToken(str);
    sprintf(str, "%f", scaleMin);
    addToken(str);
    sprintf(str, "%f", scaleMax);
    addToken(str);
    sprintf(str, "%f", scaleFactor);
    addToken(str);
    sprintf(str, "%d", trackingEnable_);
    addToken(str);

    if (navigationMode)
    {
        navigationMode_ = new char[strlen(navigationMode) + 1];
        strcpy(navigationMode_, navigationMode);
        addToken(navigationMode_);
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRSetTrackingParamsMsg::coGRSetTrackingParamsMsg(const char *msg)
    : coGRMsg(msg)
{
    vector<string> tokens = getAllTokens();
    string tmp;
    int tmpint;

    tmp = tokens[0];
    sscanf(tmp.c_str(), "%d", &tmpint);
    rotatePoint_ = tmpint != 0;
    tmp = tokens[1];
    sscanf(tmp.c_str(), "%d", &tmpint);
    rotatePointShown_ = tmpint != 0;
    tmp = tokens[2];
    sscanf(tmp.c_str(), "%f", &rotationPointSize_);
    tmp = tokens[3];
    sscanf(tmp.c_str(), "%f", &rotatePointX_);
    tmp = tokens[4];
    sscanf(tmp.c_str(), "%f", &rotatePointY_);
    tmp = tokens[5];
    sscanf(tmp.c_str(), "%f", &rotatePointZ_);
    tmp = tokens[6];
    sscanf(tmp.c_str(), "%d", &tmpint);
    rotateAxis_ = tmpint != 0;
    tmp = tokens[7];
    sscanf(tmp.c_str(), "%f", &rotateAxisX_);
    tmp = tokens[8];
    sscanf(tmp.c_str(), "%f", &rotateAxisY_);
    tmp = tokens[9];
    sscanf(tmp.c_str(), "%f", &rotateAxisZ_);
    tmp = tokens[10];
    sscanf(tmp.c_str(), "%d", &tmpint);
    translateRestrict_ = tmpint != 0;
    tmp = tokens[11];
    sscanf(tmp.c_str(), "%f", &translateMinX_);
    tmp = tokens[12];
    sscanf(tmp.c_str(), "%f", &translateMaxX_);
    tmp = tokens[13];
    sscanf(tmp.c_str(), "%f", &translateMinY_);
    tmp = tokens[14];
    sscanf(tmp.c_str(), "%f", &translateMaxY_);
    tmp = tokens[15];
    sscanf(tmp.c_str(), "%f", &translateMinZ_);
    tmp = tokens[16];
    sscanf(tmp.c_str(), "%f", &translateMaxZ_);
    tmp = tokens[17];
    sscanf(tmp.c_str(), "%f", &translateFactor_);
    tmp = tokens[18];
    sscanf(tmp.c_str(), "%d", &tmpint);
    scaleRestrict_ = tmpint != 0;
    tmp = tokens[19];
    sscanf(tmp.c_str(), "%f", &scaleMin_);
    tmp = tokens[20];
    sscanf(tmp.c_str(), "%f", &scaleMax_);
    tmp = tokens[21];
    sscanf(tmp.c_str(), "%f", &scaleFactor_);
    trackingEnable_ = tmpint != 0;
    tmp = tokens[22];
    sscanf(tmp.c_str(), "%d", (int *)&trackingEnable_);

    navigationMode_ = new char[strlen(tokens[23].c_str()) + 1];
    strcpy(navigationMode_, tokens[23].c_str());
}
