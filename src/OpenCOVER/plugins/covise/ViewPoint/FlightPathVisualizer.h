/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VIEWPOINT_VISUALIZER_H
#define _VIEWPOINT_VISUALIZER_H

#include "ViewDesc.h"

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
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include "BezierCurveVisualizer.h"
#include <list>

using namespace osg;

class FlightPathVisualizer
{
private:
    list<ViewDesc *> vpList;
    std::vector<BezierCurveVisualizer *> bezierCurveVis;

    ref_ptr<Geode> cameraGeode;
    ref_ptr<Switch> cameraSwitchNode;
    ref_ptr<Switch> flightpathSwitchNode;
    ref_ptr<MatrixTransform> flightPathDCS;
    ref_ptr<MatrixTransform> lineDCS;
    ref_ptr<MatrixTransform> cameraDCS;

    ref_ptr<osg::Geometry> line1;
    ref_ptr<osg::Geometry> line2;
    ref_ptr<osg::Geometry> line3;
    ref_ptr<osg::Geometry> line4;

    ref_ptr<Vec3Array> lineEyetoLeftDown;
    ref_ptr<Vec3Array> lineEyetoRightDown;
    ref_ptr<Vec3Array> lineEyetoRightUp;
    ref_ptr<Vec3Array> lineEyetoLeftUp;

    ref_ptr<Vec3Array> cameraPlaneCoords;
    ref_ptr<Vec3Array> cameraBorderCoords;
    ref_ptr<Geometry> cameraGeoset;
    ref_ptr<Geometry> cameraPlaneGeoset;
    ref_ptr<Geometry> cameraPlaneBorderGeoset;

    ref_ptr<StateSet> cameraPlaneGeoset_state;
    ref_ptr<StateSet> cameraGeoset_state;
    ref_ptr<StateSet> cameraPlaneBorderGeoset_state;

    ref_ptr<StateSet> line1_state;
    ref_ptr<StateSet> line2_state;
    ref_ptr<StateSet> line3_state;
    ref_ptr<StateSet> line4_state;

    ushort *cameraCoordIndexList;
    Vec3 eyepoint;

    bool shiftFlightpathToEyePoint;
    float scale;

    std::vector<ViewDesc *> *viewpoints;

    void drawLine(MatrixTransform *master, Vec3 point1, Vec3 point2, Vec4 color);
    void casteljau(MatrixTransform *master, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, int steps);
    void loadUnlightedGeostate(ref_ptr<StateSet> state);

    void drawCurve(MatrixTransform *master, std::vector<Vec3> &controlPoints);
    Vec3 computePointOnCurve(std::vector<Vec3> controlPoints, float t);
    double bernstein(double n, double i, float t);
    double binomKoeff(float n, float k);
    double factorial(double k);
    Vec3 casteljauAdvanced(std::vector<Vec3> &points, float t);
    void visualizeCasteljau(MatrixTransform *master, std::vector<Vec3> &points, float t);

public:
    FlightPathVisualizer(const FlightPathVisualizer &cc);
    FlightPathVisualizer(coVRPluginSupport *cover, std::vector<ViewDesc *> *viewPoints);
    ~FlightPathVisualizer();

    void addViewpoint(ViewDesc *viewDesc);
    void removeViewpoint(ViewDesc *viewDesc);
    void createCameraGeometry(ViewDesc *viewDesc);
    void deleteCameraGeometry();

    // show Camera
    //	void showCamera(bool state);

    void showFlightpath(bool state);

    void updateCamera(Matrix cameraMatrix);
    void updateDrawnCurve();
    void shiftFlightpath(bool state);
    MatrixTransform *getCameraDCS()
    {
        return cameraDCS;
    }
};
#endif
