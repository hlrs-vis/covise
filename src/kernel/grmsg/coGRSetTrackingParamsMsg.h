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

    bool isRotatePoint() const
    {
        return rotatePoint_;
    };
    bool isRotatePointVisible() const
    {
        return rotatePointShown_;
    };
    float getRotationPointSize() const
    {
        return rotationPointSize_;
    };
    float getRotatePointX() const
    {
        return rotatePointX_;
    };
    float getRotatePointY() const
    {
        return rotatePointY_;
    };
    float getRotatePointZ() const
    {
        return rotatePointZ_;
    };
    bool isRotateAxis() const
    {
        return rotateAxis_;
    };
    float getRotateAxisX() const
    {
        return rotateAxisX_;
    };
    float getRotateAxisY() const
    {
        return rotateAxisY_;
    };
    float getRotateAxisZ() const
    {
        return rotateAxisZ_;
    };

    bool isTranslateRestrict() const
    {
        return translateRestrict_;
    };
    float getTranslateMinX() const
    {
        return translateMinX_;
    };
    float getTranslateMaxX() const
    {
        return translateMaxX_;
    };
    float getTranslateMinY() const
    {
        return translateMinY_;
    };
    float getTranslateMaxY() const
    {
        return translateMaxY_;
    };
    float getTranslateMinZ() const
    {
        return translateMinZ_;
    };
    float getTranslateMaxZ() const
    {
        return translateMaxZ_;
    };
    float getTranslateFactor() const
    {
        return translateFactor_;
    };

    bool isScaleRestrict() const
    {
        return scaleRestrict_;
    };
    float getScaleMin() const
    {
        return scaleMin_;
    };
    float getScaleMax() const
    {
        return scaleMax_;
    };
    float getScaleFactor() const
    {
        return scaleFactor_;
    };

    bool isTrackingOn() const
    {
        return trackingEnable_;
    };
    const char *getNavigationMode() const
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
