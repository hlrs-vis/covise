/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetTrackingParamsMsg - message to set the params from the       ++
// ++                            tracking manager                         ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETTRACKINGPARAMSMSG_H
#define COGRSETTRACKINGPARAMSMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetTrackingParamsMsg : public coGRMsg
{
public:
    coGRSetTrackingParamsMsg(
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
        const char *navigationMode);
    coGRSetTrackingParamsMsg(const char *msg);

    bool isRotatePoint()
    {
        return rotatePoint_;
    };
    bool isRotatePointVisible()
    {
        return rotatePointShown_;
    };
    float getRotationPointSize()
    {
        return rotationPointSize_;
    };
    float getRotatePointX()
    {
        return rotatePointX_;
    };
    float getRotatePointY()
    {
        return rotatePointY_;
    };
    float getRotatePointZ()
    {
        return rotatePointZ_;
    };
    bool isRotateAxis()
    {
        return rotateAxis_;
    };
    float getRotateAxisX()
    {
        return rotateAxisX_;
    };
    float getRotateAxisY()
    {
        return rotateAxisY_;
    };
    float getRotateAxisZ()
    {
        return rotateAxisZ_;
    };

    bool isTranslateRestrict()
    {
        return translateRestrict_;
    };
    float getTranslateMinX()
    {
        return translateMinX_;
    };
    float getTranslateMaxX()
    {
        return translateMaxX_;
    };
    float getTranslateMinY()
    {
        return translateMinY_;
    };
    float getTranslateMaxY()
    {
        return translateMaxY_;
    };
    float getTranslateMinZ()
    {
        return translateMinZ_;
    };
    float getTranslateMaxZ()
    {
        return translateMaxZ_;
    };
    float getTranslateFactor()
    {
        return translateFactor_;
    };

    bool isScaleRestrict()
    {
        return scaleRestrict_;
    };
    float getScaleMin()
    {
        return scaleMin_;
    };
    float getScaleMax()
    {
        return scaleMax_;
    };
    float getScaleFactor()
    {
        return scaleFactor_;
    };

    bool isTrackingOn()
    {
        return trackingEnable_;
    };
    const char *getNavigationMode()
    {
        return navigationMode_;
    };

private:
    // rotate
    bool rotatePoint_;
    bool rotatePointShown_;
    float rotationPointSize_;
    float rotatePointX_;
    float rotatePointY_;
    float rotatePointZ_;
    bool rotateAxis_;
    float rotateAxisX_;
    float rotateAxisY_;
    float rotateAxisZ_;
    // translate
    bool translateRestrict_;
    float translateMinX_;
    float translateMaxX_;
    float translateMinY_;
    float translateMaxY_;
    float translateMinZ_;
    float translateMaxZ_;
    float translateFactor_;
    // scale
    bool scaleRestrict_;
    float scaleMin_;
    float scaleMax_;
    float scaleFactor_;
    // navigation
    bool trackingEnable_;
    char *navigationMode_;
};
}
#endif
