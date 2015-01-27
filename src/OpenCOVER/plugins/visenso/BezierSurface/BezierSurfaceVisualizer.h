/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * BezierSurfaceVisualizer.h
 *
 *  Created on: Dec 13, 2010
 *      Author: tm_te
 */

#ifndef BEZIERSURFACEVISUALIZER_H_
#define BEZIERSURFACEVISUALIZER_H_

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

class BezierSurfaceVisualizer
{

private:
    ref_ptr<MatrixTransform> bezierDCS;
    ref_ptr<MatrixTransform> lineDCS;

    std::vector<Vec3> controlPoints;
    int n;
    int m;

    bool showCasteljauGeom;
    bool showSurfaceGeom;
    bool showControlPolygonGeom;
    double parameterV;
    double parameterU;

    void drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color, float linewidth);
    void drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color);
    void drawPatch(ref_ptr<MatrixTransform> master, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, Vec4 color);
    Vec3 computeNormal(Vec3 p1, Vec3 p2, Vec3 p3);
    void loadUnlightedGeostate(ref_ptr<StateSet> state);
    void drawSurface(ref_ptr<MatrixTransform> master);
    double bernstein(double n, double i, double t);
    double binomKoeff(double n, double k);
    double factorial(double k);
    Vec3 casteljauAproximation(std::vector<Vec3> points, double t);
    Vec3 visualizeCasteljauCurve(ref_ptr<MatrixTransform> master, std::vector<Vec3> points, double t);
    Vec3 visualizeCasteljauCurve(ref_ptr<MatrixTransform> master, std::vector<Vec3> points, double t, int colorOffset);
    void visualizeCasteljau(ref_ptr<MatrixTransform> master);

public:
    Vec3 computePointOnCurve(double u, double v);
    BezierSurfaceVisualizer(const BezierSurfaceVisualizer &cc);
    BezierSurfaceVisualizer(ref_ptr<MatrixTransform> master);
    ~BezierSurfaceVisualizer();
    void showSurface(bool state);
    void showControlPolygon(bool state);
    void showCasteljau(bool state);
    void updateGeometry();
    void addControlPoint(Vec3 newPoint);
    void setN(int newN);
    void setM(int newM);
    int getN();
    int getM();
    void updateControlPoints(std::vector<Vec3> newPointList, int n, int m);
    void removeAllControlPoints();
    void changeControlPoint(Vec3 changePoint, int index);
    void setParameterU(double newU);
    void setParameterV(double newV);
    void degreeElevation(char direction);
    std::vector<Vec3> degreeReductionForward(char direction);
    std::vector<Vec3> degreeReductionBackward(char direction);
    void degreeReductionForest(char direction);
    void degreeReductionFarin(char direction);
    std::vector<Vec3> getAllControlPoints();
};

#endif /* BEZIERSURFACEVISUALIZER_H_ */
