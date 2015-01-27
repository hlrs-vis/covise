/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VIEWPOINTINTERPOLATOR_H_
#define VIEWPOINTINTERPOLATOR_H_

#include <stdlib.h>

//#include <Performer/pr/pfLinMath.h>
//#include <cover/coVRModuleSupport.h>
#include "ViewDesc.h"
//#include "FlightPathVisualizer.h"

//#include <Performer/pf/pfScene.h>
//#include <Performer/pf/pfDCS.h>
//#include <Performer/pf/pfGeode.h>
//#include <Performer/pr/pfGeoSet.h>
//#include <Performer/pf/pfGroup.h>
//#include <Performer/pf/pfNode.h>
//#include <Performer/pr/pfMaterial.h>
//#include <Performer/pr/pfLinMath.h>

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ClipNode>
#include <osg/ClipNode>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/LineWidth>
#include <osg/Material>
#include <osg/Vec3>

class Interpolator
{
public:
    enum TranslationMode
    {
        LINEAR_TRANSLATION,
        BEZIER
    };
    enum RotationMode
    {
        QUATERNION,
        FOLLOWPATH
    };
    enum EasingFunction
    {
        LINEAR_EASING,
        QUADRIC_IN,
        QUADRIC_OUT,
        QUADRIC_IN_OUT,
        CUBIC_IN,
        CUBIC_OUT,
        CUBIC_IN_OUT,
        QUARTIC_IN,
        QUARTIC_OUT,
        QUARTIC_IN_OUT
    };

private:
    Vec3 casteljau(Vec3 points[], float lambda, int depth);
    TranslationMode translationMode;

public:
    Interpolator();
    virtual ~Interpolator();

    float interpolateScale(float startScale, ViewDesc *destination,
                           float lambda);
    Matrix interpolateRotation(Matrix startMatrix, Vec3 tangentOut, double startScale, double destScale, Matrix destinationMat,
                               Vec3 tangentIn, float lambda, RotationMode rotationMode);
    Matrix interpolateRotation(coCoord startCoord, Vec3 tangentOut, double startScale, ViewDesc *destination,
                               Vec3 tangentIn, float lambda, RotationMode rotationMode);
    Matrix interpolateTranslation(
        Matrix startMatrix, Vec3 tangentOut, double startScale, Matrix destinationMatrix, double destinationScale, Vec3 tangentIn,
        float lambda, float interpolatedScale, TranslationMode translationMode);
    Matrix interpolateTranslation(
        coCoord startCoord, Vec3 tangentOut, double startScale, coCoord destinationCoord, double destinationScale, Vec3 tangentIn,
        float lambda, float interpolatedScale, TranslationMode translationMode);
    float easingFunc(EasingFunction e, float t, float b, float c, float d);

private:
    Vec3 interpolateTranslationLinear(Vec3 points[], float lambda);
    Vec3 interpolateTranslationBezier(Vec3 points[], float lambda);
};

#endif /*VIEWPOINTINTERPOLATOR_H_*/
