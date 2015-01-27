/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * BezierPlugin.h
 *
 *  Created on: Dec 3, 2010
 *      Author: tm_te
 */

#ifndef BEZIERCURVEVISUALIZER_H_
#define BEZIERCURVEVISUALIZER_H_

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ClipNode>
#include <osg/ClipNode>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/LineWidth>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec3>
//#include <cover/coVRModuleSupport.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coInteractor.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

using namespace osg;
using namespace opencover;
using namespace covise;

class BezierCurveVisualizer
{

public:
    enum Computation
    {
        EXACT_COMPUTATION,
        APROXIMATION,
        CUBIC_APROXIMATION
    };

private:
    ref_ptr<MatrixTransform> bezierDCS;
    ref_ptr<MatrixTransform> lineDCS;

    std::vector<Vec3> controlPoints;

    bool showCasteljauGeom;
    bool showCurveGeom;
    bool showControlPolygonGeom;
    bool showTangentBeginGeom;
    bool showTangentEndGeom;
    double t;

    Computation computation;

    void drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color, float linewidth);
    void drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color);
    void cubicCasteljauAproximation(ref_ptr<MatrixTransform> master, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, int steps);
    void loadUnlightedGeostate(ref_ptr<StateSet> state);
    void drawCurve(ref_ptr<MatrixTransform> master);
    double bernstein(double n, double i, double t);
    double binomKoeff(double n, double k);
    double factorial(double k);
    Vec3 casteljauAproximation(std::vector<Vec3> points, double t);
    void visualizeCasteljau(ref_ptr<MatrixTransform> master, double t);

public:
    BezierCurveVisualizer(const BezierCurveVisualizer &cc);
    BezierCurveVisualizer(ref_ptr<MatrixTransform> master, Computation computation);
    ~BezierCurveVisualizer();

    Vec3 computePointOnCurve(double t);
    void showCurve(bool state);
    void showControlPolygon(bool state);
    void showCasteljau(bool state);
    void showTangentBegin(bool state);
    void showTangentEnd(bool state);
    void updateGeometry();
    void addControlPoint(Vec3 newPoint);
    void addVectorOfControlPoints(std::vector<Vec3> newPoints);
    void removeControlPoint();
    void removeAllControlPoints();
    void updateControlPoints(std::vector<Vec3> newPointList);
    void changeControlPoint(Vec3 changePoint, int index);
    void setT(double newT);
    void degreeElevation();
    std::vector<Vec3> degreeReductionForward();
    std::vector<Vec3> degreeReductionBackward();
    void degreeReductionForest();
    void degreeReductionFarin();
    std::vector<Vec3> getAllControlPoints();
};

#endif /* BEZIERCURVEVISUALIZER_H_ */
