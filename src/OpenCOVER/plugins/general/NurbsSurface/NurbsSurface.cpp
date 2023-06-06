/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: NurbsSurface OpenCOVER Plugin (draws a NurbsSurface)        **
 **                                                                          **
 **                                                                          **
 ** Author: F.Karle/ K.Ahmann                                                **
 **                                                                          **
 ** History:                                                                 **
 ** December 2017  v1                                                        **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "NurbsSurface.h"
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <cover/coVRCommunication.h>

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osg/Vec4>
#include <osg/PolygonStipple>
#include <osg/TriangleFunctor>
#include <osg/LightModel>
#include <cover/VRSceneGraph.h>
#include <osgDB/WriteFile>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/coVRSceneView.h>
#include <cover/coVRMSController.h>
#include <cover/VRWindow.h>
#include <cover/coVRTui.h>
#include <cover/coVRRenderer.h>
#include <util/coFileUtil.h>
#include <osgViewer/Renderer>

#include <PluginUtil/PluginMessageTypes.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Button.h>

#include <grmsg/coGRSnapshotMsg.h>
#include <net/tokenbuffer.h>

#include <osgDB/WriteFile>
#include <osg/Camera>

#include <iostream>
#include <sstream>
#include <string>

#include "../PointCloud/Points.h"
#include "../PointCloud/FileInfo.h"

#include <boost/algorithm/string.hpp>
#include "sisl.h"
#include "sisl_aux/sisl_aux.h"

#include "alglib/interpolation.h"
#include "alglib/stdafx.h"
#include "alglib/statistics.h"
#include "alglib/linalg.h"

static const int DEFAULT_REF = 30000000; 
static const int DEFAULT_MAX_REF = 100;

using namespace opencover;
using namespace std;
using namespace grmsg;
//using namespace alglib;

using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::coDirectory;
  

void NurbsSurface::initUI()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- NurbsSurface::initUI\n");

    NurbsSurfaceMenu = new ui::Menu("NurbsSurface", this);
    NurbsSurfaceMenu->setText("NurbsSurface");

    currentSurfaceLabel = new ui::Label(NurbsSurfaceMenu,"currentSurfaceIndex");
    currentSurfaceLabel->setText("Test");

    newSurface = new ui::Action(NurbsSurfaceMenu, "createNewSurface");
    newSurface->setText("create new surface");
    newSurface->setCallback([this](){
      createNewSurface();
      updateUI();
    });

    saveButton_ = new ui::Action(NurbsSurfaceMenu, "SaveFile");
    saveButton_->setText("SaveFile");
    saveButton_->setCallback([this](){
        saveFile("test.obj");
    });

    surfaceSelectionSlider = new ui::Slider(NurbsSurfaceMenu, "selected_Surface");
    surfaceSelectionSlider->setIntegral(true);
    surfaceSelectionSlider->setText("Index of selected surface");
    surfaceSelectionSlider->setBounds(0,surfaces.size()-1);
    surfaceSelectionSlider->setCallback([this](int val, bool released)
    {
        currentSurface = &surfaces[val];
        selectionSetMessage();
        updateUI();
    }
    );

    selectionIsBoundaryButton = new ui::Button(NurbsSurfaceMenu, "SelectBoundary");
    selectionIsBoundaryButton->setText("Select Boundary");
    selectionIsBoundaryButton->setCallback([this](bool state){
        if (state)
        {
        setSelectionIsBoundary(true);
        }
        else
        {
        setSelectionIsBoundary(false);
        }
    });

    selectionParameters = new ui::Group(NurbsSurfaceMenu,"selectionParameters");
    selectionParameters->setText("parameters of selected surface");

    orderUSlider = new ui::Slider(selectionParameters, "order_U");
    orderVSlider = new ui::Slider(selectionParameters, "order_V");

    orderUSlider->setIntegral(true);
    orderUSlider->setText("Order U");
    orderUSlider->setBounds(2, 4);
    orderUSlider->setValue(currentSurface->order_U);;
    orderUSlider->setCallback([this](int val, bool released)
    {
        currentSurface->destroy();
        currentSurface->order_U=val;
        currentSurface->updateSurface();
    }
    );

    orderVSlider->setIntegral(true);
    orderVSlider->setText("Order V");
    orderVSlider->setBounds(2, 4);
    orderVSlider->setValue(currentSurface->order_V);;
    orderVSlider->setCallback([this](int val, bool released)
    {
        currentSurface->destroy();
        currentSurface->order_V=val;
        currentSurface->updateSurface();
    }
    );

    numEdgeSectorsSlider = new ui::Slider(selectionParameters, "edge_sectors");
    numEdgeSectorsSlider->setIntegral(true);
    numEdgeSectorsSlider->setText("Number of sectors for edge interpolation");
    numEdgeSectorsSlider->setBounds(2,10);
    numEdgeSectorsSlider->setValue(currentSurface->numEdgeSectors);
    numEdgeSectorsSlider->setCallback([this](int val, bool released)
    {
        currentSurface->destroy();
        currentSurface->numEdgeSectors=val;
        currentSurface->updateSurface();
    }
    );
}

void NurbsSurface::saveFile(const std::string &fileName)
{
        osgDB::writeNodeFile(*currentSurface->geode, fileName.c_str());
}

bool NurbsSurface::init()
{
        if (cover->debugLevel(3))
                fprintf(stderr, "\n--- NurbsSurface::init\n");
        //initialize first surface
        createNewSurface();
        initUI();
        updateUI();
        //createRBFModel();
        /*for (std::vector<surfaceInfo>::iterator it=surfaces.begin(); it != surfaces.end(); it++)
        {
            fprintf(stderr, "\n--- create model\n");
            it->createRBFModel();
        }*/
        return true;
}

osg::ref_ptr<osg::Geode> NurbsSurface::surfaceInfo::computeSurface(double* points)
{
    SISLSurf* result_surf = 0;
    int jstat = 0;

    s1537(points,       // pointer to the array of points to interpolate
          num_points_u, // number of interpolating points along the 'u' parameter
          num_points_v, // number of interpolating points along the 'v' parameter
          dim,          // dimension of the Euclidean space
          u_par,        // pointer to the 'u' parameter values of the points
          v_par,        // pointer to the 'v' parameter values of the points
          0,            // no additional condition along edge 1
          0,            // no additional condition along edge 2
          0,            // no additional condition along edge 3
          0,            // no additional condition along edge 4
          order_U,   // the order of the generated surface in the 'u' parameter
          order_V,   // the order of the generated surface in the 'v' parameter
          1,            // open surface in the u direction
          1,            // open surface in the v direction
          &result_surf, // the generated surface
          &jstat);      // status variable
    if (jstat < 0) {
        throw runtime_error("Error occured inside call to SISL routine s1537.");
    } else if (jstat > 0) {
        cerr << "WARNING: warning occured inside call to SISL routine s1537. \n";
    }

    int ref=DEFAULT_REF; // Number of new knots between old ones.
    int maxref=DEFAULT_MAX_REF; // Maximal number of coeffs in any given direction.

    lower_degree_and_subdivide(&result_surf, ref, maxref);
    double *normal;
    compute_surface_normals(result_surf, &normal);

    // create the Geode (Geometry Node) to contain all our osg::Geometry objects.
    geode = new osg::Geode();

    // create Geometry object to store all the vertices and lines primtive
    osg::Geometry* polyGeom = new osg::Geometry();

    osg::Vec3Array* vertices = new osg::Vec3Array;
    osg::Vec3Array* normals = new osg::Vec3Array;
    osg::Vec4Array* colors = new osg::Vec4Array;
    osg::Vec4 _color;
    _color.set(1.0, 0.0, 0.0, 1.0);

    for (int j=0; j<result_surf->in1-1; j++)
    {
        int vertexBegin=vertices->size();
        for (int k=0; k<result_surf->in2; k++)
        {
            int p=3*(j+k*result_surf->in1);
            vertices->push_back(rotationMatrixToWorld * osg::Vec3(result_surf->ecoef[p+0], result_surf->ecoef[p+1], result_surf->ecoef[p+2]));
            normals->push_back(rotationMatrixToWorld * osg::Vec3(normal[p+0], normal[p+1], normal[p+2]));
            colors->push_back(_color);
            p+=3;
            vertices->push_back(rotationMatrixToWorld * osg::Vec3(result_surf->ecoef[p+0], result_surf->ecoef[p+1], result_surf->ecoef[p+2]));
            normals->push_back(rotationMatrixToWorld * osg::Vec3(normal[p+0], normal[p+1], normal[p+2]));
            colors->push_back(_color);
        }
        int vertexEnd=vertices->size()-vertexBegin;
        polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,vertexBegin,vertexEnd));
    }

    // pass the created vertex array to the points geometry object
    polyGeom->setVertexArray(vertices);

    // use the color array.
    polyGeom->setColorArray(colors);
    polyGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    // use the normal array.
    polyGeom->setNormalArray(normals);
    polyGeom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    //TriangleStrip
    geode->addDrawable(polyGeom);

    // stateSet
    osg::StateSet* stateSet = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);//polyGeom->getOrCreateStateSet();
    /*osg::Material* matirial = new osg::Material;
          matirial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
          matirial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0.3, 0, 1));
          matirial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0.3, 0, 1));
          matirial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0.3, 0, 1));
          matirial->setShininess(osg::Material::FRONT_AND_BACK, 10.0f);
          stateSet->setAttributeAndModes (matirial,osg::StateAttribute::ON); */
    osg::LightModel* ltModel = new osg::LightModel;
    ltModel->setTwoSided(true);
    stateSet->setAttribute(ltModel);
    stateSet->setMode( GL_CULL_FACE, osg::StateAttribute::OFF );
    polyGeom->setStateSet(stateSet);

    freeSurf(result_surf);

    // add the points geomtry to the geode.
    cover->getObjectsRoot()->addChild(geode.get());

    // send back surface to PointCloudPlugin for intersection testing and pointcloud manipulation

    return geode.get();
}

NurbsSurface::NurbsSurface()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("NurbsSurface", cover->ui)

{
    //updateSurface();

}

bool NurbsSurface::surfaceInfo::destroy()
{
    cover->getObjectsRoot()->removeChild(geode.get());
    return true;
}


// this is called if the plugin is removed at runtime
NurbsSurface::~NurbsSurface()
{
    fprintf(stderr, "Goodbye\n");
    for (std::vector<surfaceInfo>::iterator it=surfaces.begin(); it != surfaces.end(); it++)
    {
        cover->getObjectsRoot()->removeChild(it->splinePointsGroup.get());
    }
}

void NurbsSurface::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::NurbsSurfacePointMsg)
    {
        vector<pointSelection> *selectedPoints = (vector<pointSelection> *)buf;
        currentSurface->receivedPoints.clear();
        currentSurface->receivedBoundaryPoints.clear();
        int numBoundaryPoints = 0;
        for (vector<pointSelection>::const_iterator iter=selectedPoints->begin(); iter!=selectedPoints->end(); iter++)
        {
            if (iter->selectionIndex==currentSurface->surfaceIndex)
            {
				Vec3 newSelectedPoint = iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].coordinates;
                if (iter->isBoundaryPoint)
                {
                    currentSurface->receivedBoundaryPoints.push_back(newSelectedPoint);
                }
                else
                {
                    currentSurface->receivedPoints.push_back(newSelectedPoint);
                }
            }
        }
        fprintf(stderr, "Points received %zi\n", currentSurface->receivedPoints.size());
        for (auto iter=currentSurface->receivedPoints.begin(); iter!=currentSurface->receivedPoints.end(); iter++)
        {
            fprintf(stderr, "Point x: %f, y: %f, z: %f \n", iter->x(), iter->y(), iter->z());
        }
        fprintf(stderr, "%zi points are boundary points \n", currentSurface->receivedBoundaryPoints.size());
        currentSurface->updateSurface();
    }
}

void NurbsSurface::surfaceInfo::updateSurface()
{
    while (splinePointsGroup->getNumChildren()>0)
    {
        splinePointsGroup->removeChildren(0,1);
    }
    transformMatrices.clear();
    if (receivedPoints.size()>20)
    {
        updateModel();
        fprintf(stderr, "NurbsSurface::surfaceInfo::updateSurface  %zi points are boundary points \n", receivedBoundaryPoints.size());
        if (receivedBoundaryPoints.size()== 4)
        {
            std::vector<curveInfo> boundaryCurves;
            calcEdgeIntersectionsByPoints(boundaryCurves);
            calcSamplingPoints(boundaryCurves);
        }
        else
        {
            calcEdges();
            calcSamplingPoints();
        }
        resize();
    }
}

void NurbsSurface::surfaceInfo::calcEdgeIntersectionsByPoints(std::vector<curveInfo>& curves)
{
    bool intersectionSuccess = true;
    edgeColor =osg::Vec4f(0.0, 0.1, 0.1, 1.0f);
    for (auto it = receivedBoundaryPointsRotated.begin(); it != receivedBoundaryPointsRotated.end(); it++)
    {
        auto nx = std::next(it, 1);
        if (nx == receivedBoundaryPointsRotated.end())
        {
            nx=receivedBoundaryPointsRotated.begin();
        }
        fprintf(stderr,"Point it x: %f  y: %f  z: %f\n",it->x(),it->y(),it->z());
        fprintf(stderr,"Point nx x: %f  y: %f  z: %f\n",nx->x(),nx->y(),it->z());
        curveInfo boundaryCurve;
        edgeByPoints(receivedPointsRotated, *it, *nx, boundaryCurve);
        edgeColor +=osg::Vec4f(0.0, 0.1, 0.1, 1.0f);
        curves.push_back(boundaryCurve);
    }
    for (auto it = curves.begin();it !=curves.end(); it++)
    {
        auto nx = std::next(it, 1);
        if (nx == curves.end())
        {
            nx=curves.begin();
        }
        intersectionSuccess = curveCurveIntersection(it->curve, it->endPar, nx->curve, nx->startPar);
    }
}

void NurbsSurface::surfaceInfo::calcSamplingPoints(std::vector<curveInfo>& curves)
{
    if (curves.size()==4)
    {
        int pointsNewSize=num_points_u*num_points_v*3;
        double *pointsNew = new double[pointsNewSize];

        int k=0;

        vector<vector<double> > corners;
        for (auto it = curves.begin();it !=curves.end(); it++)
        {
            vector<double> corner = evaluateCurveAtParam(*it, 0.0);
            corners.push_back(corner);
        }

        for (int i=0; i!=num_points_u; i++)
        {
            double paramFactor0=1.0/(num_points_u-1)*i;
            for (int j=0; j!=num_points_v;j++)
            {
                double paramFactor1=1.0/(num_points_v-1)*j;
                //calculate the linear interpolation between edge 0 and 2 as if edges 1 and 3 were straight lines
                vector<double> linearPointBetweenEdge0and2 = paramFactor1 * evaluateCurveAtParam(curves[0],paramFactor0) + (1.0-paramFactor1) * evaluateCurveAtParam(curves[2],(1-paramFactor0));
                //then calculate compensation on edges 1 and 3 as difference between straight line and actual point on line
                vector<double> shiftOnEdge1 = paramFactor0 * (evaluateCurveAtParam(curves[1],paramFactor1) - (corners[1] +(corners[2]-corners[1])*paramFactor1));
                vector<double> shiftOnEdge3 = (1.0-paramFactor0) * (evaluateCurveAtParam(curves[3],(1.0 - paramFactor1)) - (corners[0]+(corners[3]-corners[0])*paramFactor1));
                vector<double> interpolatedPoint = linearPointBetweenEdge0and2 + shiftOnEdge1 + shiftOnEdge3;

                pointsNew[k++]=interpolatedPoint[0];
                pointsNew[k++]=interpolatedPoint[1];

                real_1d_array x;
                x.setlength(2);
                x[0]=interpolatedPoint[0];
                x[1]=interpolatedPoint[1];
                real_1d_array y;
                y.setlength(1);
                rbfcalc(model,x,y);
                pointsNew[k++]=y[0];
            }
        }
        destroy();
        computeSurface(pointsNew);
        delete[] pointsNew;
    }
    else
    {
        fprintf(stderr, "4 curves needed, %zi curves given \n", curves.size());
    }
}

void NurbsSurface::surfaceInfo::calcSamplingPoints()
{
    int pointsNewSize=num_points_u*num_points_v*3;
    double *pointsNew = new double[pointsNewSize];//num_points_u*num_points_v*3];
    vector<double> pointTempUpper(2);
    vector<double> pointTempLower(2);
    vector<double> pointTempRight(2);
    vector<double> pointTempLeft(2);
    vector<double> pointTempRightZero(2);
    vector<double> pointTempLeftZero(2);
    vector<double> pointTempRightOne(2);
    vector<double> pointTempLeftOne(2);
    vector<double> interpolatedPoint(2);
    evaluateCurveAtParam(left,1.0,pointTempLeftOne);
    evaluateCurveAtParam(right,1.0,pointTempRightOne);
    evaluateCurveAtParam(left,0.0,pointTempLeftZero);
    evaluateCurveAtParam(right,0.0,pointTempRightZero);
    int k=0;
    for (int i=0; i!=num_points_u; i++)
    {
        double paramFactor0=1.0/(num_points_u-1)*i;
        fprintf(stderr,"paramFactor0: %f \n",paramFactor0);
        for (int j=0; j!=num_points_v;j++)
        {
            double paramFactor1=1.0/(num_points_v-1)*j;
            fprintf(stderr,"paramFactor1: %f \n",paramFactor1);
            //get a linear interpolation between upper and lower curve at paramFactor0
            evaluateCurveAtParam(upper,paramFactor0,pointTempUpper);
            evaluateCurveAtParam(lower,paramFactor0,pointTempLower);
            vector<double> pointTempUpperLower = pointTempUpper+(pointTempLower-pointTempUpper)*paramFactor1;
            //get points on left and right boundary according to paraFactor1
            evaluateCurveAtParam(left,paramFactor1,pointTempLeft);
            evaluateCurveAtParam(right,paramFactor1,pointTempRight);
            vector<double> shiftAccordingtoRightBoundary = paramFactor0*(pointTempRight -(pointTempRightZero+(pointTempRightOne-pointTempRightZero)*paramFactor1));
            vector<double> shiftAccordingtoLeftBoundary = (1.0-paramFactor0)*(pointTempLeft -(pointTempLeftZero+(pointTempLeftOne-pointTempLeftZero)*paramFactor1));
            //vector<double> pointTempLeftRight = (paramFactor0*(pointTempRight -(pointTempRightZero+(pointTempRightOne-pointTempRightZero)*paramFactor1))+(1.0-paramFactor0)*(pointTempLeft -(pointTempLeftZero+(pointTempLeftOne-pointTempLeftZero)*paramFactor1)));
            interpolatedPoint = pointTempUpperLower + shiftAccordingtoRightBoundary + shiftAccordingtoLeftBoundary;
            pointsNew[k++]=interpolatedPoint[0];
            pointsNew[k++]=interpolatedPoint[1];
            real_1d_array x;
            x.setlength(2);
            x[0]=interpolatedPoint[0];
            x[1]=interpolatedPoint[1];
            real_1d_array y;
            y.setlength(1);
            rbfcalc(model,x,y);
            pointsNew[k++]=y[0];
        }
    }
    if (cover->debugLevel(3))
    {
        int l=0;
        while (l!=pointsNewSize)
        {
            fprintf(stderr,"p: %i %f ",l,pointsNew[l]);
            if (l % 3 ==2)
                fprintf(stderr,"\n");
            l++;
        }
    }
    destroy();
    computeSurface(pointsNew);
    delete[] pointsNew;
}

void NurbsSurface::surfaceInfo::evaluateCurveAtParam(curveInfo& curve, double paramFactor, vector<double> &point)
{
    int jstat; // status variable
    int temp;
    s1227(curve.curve,0,curve.startPar+paramFactor*(curve.endPar-curve.startPar),&temp,&point[0],&jstat);
}

vector<double> NurbsSurface::surfaceInfo::evaluateCurveAtParam(curveInfo& curve, double paramFactor)
{
    vector<double> result(2);
    int jstat; // status variable
    int temp;
    s1227(curve.curve,0,curve.startPar+paramFactor*(curve.endPar-curve.startPar),&temp,&result[0],&jstat);
    return result;
}

bool NurbsSurface::surfaceInfo::calcEdges()
{
    edge(receivedPointsRotated,0,1,1,upper);
    edge(receivedPointsRotated,1,0,1,right);
    edge(receivedPointsRotated,0,1,-1,lower);
    edge(receivedPointsRotated,1,0,-1,left);
    if (upper.curve && lower.curve && left.curve && right.curve)
    {
        curveCurveIntersection(upper.curve, upper.endPar, right.curve, right.endPar);
        curveCurveIntersection(lower.curve, lower.endPar, right.curve, right.startPar);
        curveCurveIntersection(lower.curve, lower.startPar, left.curve, left.startPar);
        curveCurveIntersection(upper.curve, upper.startPar, left.curve, left.endPar);
        fprintf(stderr, "upper curve end parameter %f \n", upper.endPar);
        fprintf(stderr, "upper curve start parameter %f \n", upper.startPar);
        fprintf(stderr, "right curve end parameter %f \n", right.endPar);
        fprintf(stderr, "right curve start parameter %f \n", right.startPar);
        return true;
    }
    return false;
}

bool NurbsSurface::surfaceInfo::curveCurveIntersection(SISLCurve *c1, double& c1Param, SISLCurve *c2, double& c2Param)
{
    // calculating intersection points
    double epsco = 1.0e-15; // computational epsilon
    double epsge = 1.0e-5; // geometric tolerance
    int num_int_points = 0; // number of found intersection points
    double* intpar1 = 0; // parameter values for the first curve in the intersections
    double* intpar2 = 0; // parameter values for the second curve in the intersections
    int num_int_curves = 0;   // number of intersection curves
    SISLIntcurve** intcurve = 0; // pointer to array of detected intersection curves
    int jstat; // status variable

    s1857(c1,              // first curve
          c2,              // second curve
          epsco,           // computational resolution
          epsge,           // geometry resolution
          &num_int_points, // number of single intersection points
          &intpar1,        // pointer to array of parameter values
          &intpar2,        //               "
          &num_int_curves, // number of detected intersection curves
          &intcurve,       // pointer to array of detected intersection curves.
          &jstat);

    if (jstat < 0) {
        throw runtime_error("Error occured inside call to SISL routine s1857.");
    } else if (jstat > 0) {
        std::cerr << "WARNING: warning occured inside call to SISL routine s1857. \n" << std::endl;
    }
    fprintf(stderr,"Number of intersection points detected: %i\n", num_int_points);
    if (num_int_points>0)
    {
        c1Param=*intpar1;
        c2Param=*intpar2;
        fprintf(stderr,"value %f\n",*intpar1);

        // evaluating intersection points
        vector<double> point_coords_3D(3 * num_int_points);
        int i;
        for (i = 0; i < num_int_points; ++i) {
            // calculating position, using curve 1
            // (we could also have used curve 2, which would give approximately
            // the same points).
            int temp;
            s1227(c1,         // we evaluate on the first curve
                  0,          // calculate no derivatives
                  intpar1[i], // parameter value on which to evaluate
                  &temp,      // not used for our purposes (gives parameter interval)
                  &point_coords_3D[3 * i], // result written here
                    &jstat);

            if (jstat < 0) {
                throw runtime_error("Error occured inside call to SISL routine s1227.");
            } else if (jstat > 0) {
                std::cerr << "WARNING: warning occured inside call to SISL routine s1227. \n" << std::endl;
            }
        }
        return true;
    }
    return false;
}


int NurbsSurface::surfaceInfo::edgeByPoints(std::vector<Vec3> &selectedPoints, Vec3 pointBegin, Vec3 pointEnd, curveInfo &resultCurveInfo)
{
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints\n");
    for (auto iter = selectedPoints.begin(); iter!=selectedPoints.end(); iter++)
    {
      //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints x: %f, y: %f, z: %f \n ",iter->x(), iter->y(), iter->z() );
    }

    SISLCurve *result_curve = 0;
    std::vector<Vec3> all_points;

    //rotation of points
    Vec3 edgeDirection = pointEnd - pointBegin;
    edgeDirection[2] = 0.0;
    //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints edgeDirection x: %f, y: %f, z: %f \n ",edgeDirection.x(), edgeDirection.y(), edgeDirection.z() );
    edgeDirection.normalize();
    //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints edgeDirection x: %f, y: %f, z: %f \n ",edgeDirection.x(), edgeDirection.y(), edgeDirection.z() );

    Vec3 unitAxis = Vec3(1.0, 0.0, 0.0);
    //rotationMatrix.makeRotate(edgeDirection,unitAxis);
    rotationMatrix.makeRotate(unitAxis,edgeDirection);
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints PointBegin x: %f, y: %f, z: %f \n ",pointBegin.x(), pointBegin.y(), pointBegin.z() );
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints pointEnd x: %f, y: %f, z: %f \n ",pointEnd.x(), pointEnd.y(), pointEnd.z() );
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints centroid x: %f, y: %f, z: %f \n ",centroid.x(), centroid.y(), centroid.z() );
    }

    Vec3 pointBeginLocal = rotationMatrix * pointBegin;
    Vec3 pointEndLocal = rotationMatrix * pointEnd;
    Vec3 centroidLocal = rotationMatrix * centroidRotated;
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints PointBeginLocal x: %f, y: %f, z: %f \n ",pointBeginLocal.x(), pointBeginLocal.y(), pointBeginLocal.z() );
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints pointEndLocal x: %f, y: %f, z: %f \n ",pointEndLocal.x(), pointEndLocal.y(), pointEndLocal.z() );
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints centroidLocal x: %f, y: %f, z: %f \n ",centroidLocal.x(), centroidLocal.y(), centroidLocal.z() );
    }

    if (pointBeginLocal.x() > pointEndLocal.x())
    {
        //fprintf(stderr, "Rotated by 180deg \n ");
        Matrixd rotate;
        rotate.makeRotate(DegreesToRadians(180.0),Vec3(0.0,0.0,1.0));
        rotationMatrix = rotate * rotationMatrix;
        pointBeginLocal = rotationMatrix * pointBegin;
        pointEndLocal = rotationMatrix * pointEnd;
        centroidLocal = rotationMatrix * centroidRotated;
    }
    /*
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints matrix rotate %f, %f, %f, %f \n ",rotationMatrix(0,0), rotationMatrix(0,1), rotationMatrix(0,2),rotationMatrix(0,3));
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints matrix rotate %f, %f, %f, %f \n ",rotationMatrix(1,0), rotationMatrix(1,1), rotationMatrix(1,2),rotationMatrix(1,3));
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints matrix rotate %f, %f, %f, %f \n ",rotationMatrix(2,0), rotationMatrix(2,1), rotationMatrix(2,2),rotationMatrix(2,3));
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints matrix rotate %f, %f, %f, %f \n ",rotationMatrix(3,0), rotationMatrix(3,1), rotationMatrix(3,2),rotationMatrix(3,3));

    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints PointBeginLocal x: %f, y: %f, z: %f \n ",pointBeginLocal.x(), pointBeginLocal.y(), pointBeginLocal.z() );
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints pointEndLocal x: %f, y: %f, z: %f \n ",pointEndLocal.x(), pointEndLocal.y(), pointEndLocal.z() );
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints centroidLocal x: %f, y: %f, z: %f \n ",centroidLocal.x(), centroidLocal.y(), centroidLocal.z() );
    */

    // Test if centroid is above or below
    float dx = pointEndLocal.x() - pointBeginLocal.x();
    float dy = pointEndLocal.y() - pointBeginLocal.y();
    float slope = dy/dx;
    float intercept = pointBeginLocal.y() - slope * pointBeginLocal.x();

    float yOnLine = slope * centroidLocal.x()+intercept;
    int centroidIsBelow=1;

    if (yOnLine < centroidLocal.y())
    {
        centroidIsBelow=-1;
    }

    inverseRotationMatrix.invert(rotationMatrix);

    for (auto it=selectedPoints.begin(); it!=selectedPoints.end(); it++)
    {
        all_points.push_back(rotationMatrix * *it);
    }

    //
    double sumZ = 0.0;
    for (std::vector<osg::Vec3>::const_iterator it = all_points.begin(); it !=all_points.end(); it++)
    {
        sumZ += it->_v[2];
    }
    double averageZ = sumZ/all_points.size();

    //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints averageZ : %f\n", averageZ);

    float minimum_x = pointBeginLocal.x();
    float maximum_x = pointEndLocal.x();
    float outerMinX = minimum_x;
    float outerMaxX = maximum_x;

    //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints minimum x= %f, maximum x=%f \n",minimum_x, maximum_x);

    std::vector<osg::Vec3*> maximumPointsInAllQuadrants;
    maximumPointsInAllQuadrants.resize(numEdgeSectors, nullptr);
    real_1d_array LocalXForFirstCurve;
    real_1d_array LocalYForFirstCurve;

    if (cover->debugLevel(3))
        fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints finding sectors with maximum\n");

    int numberOfSectorsWithMaximum = 0;
    for(auto it=all_points.begin(); it != all_points.end(); it++)
    {
        int j=int((it->x()-minimum_x)/(maximum_x-minimum_x)*numEdgeSectors);
        if ((j>=0) && (j<numEdgeSectors))
        {
            if (!maximumPointsInAllQuadrants[j])
            {
                maximumPointsInAllQuadrants[j] = &*it;
                numberOfSectorsWithMaximum++;
            }
            else
            {
                if ((centroidIsBelow * it->y()) > (centroidIsBelow * maximumPointsInAllQuadrants[j]->y()))
                {
                    maximumPointsInAllQuadrants[j] = &*it;
                }
            }
        }
        if (it->x() < outerMinX)
        {
            outerMinX = it->x();
        }
        if (it->x() > outerMaxX)
        {
            outerMaxX = it->x();
        }
    }

    float distMinMax = outerMaxX - outerMinX;
    outerMaxX += 0.1 * distMinMax;
    outerMinX -= 0.1 * distMinMax;

    if (cover->debugLevel(3))
    {
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints outerMinX: %f, outerMaxX: %f \n ", outerMinX, outerMaxX);
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints numberOfSectorsWithMaximum = %i\n", numberOfSectorsWithMaximum);
    }
    //initialize spline firstCurveWithMaximumPointsPerQuadrant
    ae_int_t degree = 3;

    ae_int_t info;
    barycentricinterpolant firstCurveWithMaximumPointsPerQuadrant;
    polynomialfitreport repo;

    //initialize spline curve
    barycentricinterpolant curve;
    fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints interpolating first curve\n");
    if (numberOfSectorsWithMaximum > 1)
    {
        LocalXForFirstCurve.setlength(numberOfSectorsWithMaximum);
        LocalYForFirstCurve.setlength(numberOfSectorsWithMaximum);

        int countFirstCurve = 0;

        for(std::vector<Vec3*>::iterator it = maximumPointsInAllQuadrants.begin(); it != maximumPointsInAllQuadrants.end(); it++)
        {
            if (*it)
            {
                LocalXForFirstCurve[countFirstCurve] = (*it)->x();
                LocalYForFirstCurve[countFirstCurve] = (*it)->y();
                countFirstCurve++;
            }
        }

        //fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints building first curve out of maximum points per quadrant\n");
        //built first curve out of maximum points per quadrant
        polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, firstCurveWithMaximumPointsPerQuadrant, repo);

        std::vector<osg::Vec3> pointsAboveCurve;

        //compare all points with first curve, if they are above or below
        for(auto it = all_points.begin(); it != all_points.end(); it++)
        {
            if ((it->x() > minimum_x) && (it->x() < maximum_x ))
            {
                if (it->y() > barycentriccalc(firstCurveWithMaximumPointsPerQuadrant, it->x()))
                {
                    pointsAboveCurve.push_back(*it);
                }
            }
        }
        if (cover->debugLevel(3))
        {
            fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints pointsAboveCurve = %li\n", (long)pointsAboveCurve.size());
        }
        real_1d_array LocalXForSecondCurve;
        real_1d_array LocalYForSecondCurve;

        LocalXForSecondCurve.setlength(numberOfSectorsWithMaximum + pointsAboveCurve.size());
        LocalYForSecondCurve.setlength(numberOfSectorsWithMaximum + pointsAboveCurve.size());

        int countSecondCurve = 0;

        for(int k = 0; k < maximumPointsInAllQuadrants.size(); k++)
        {
            if (!maximumPointsInAllQuadrants[k]) {
            }
            else{
                LocalXForSecondCurve[countSecondCurve] = maximumPointsInAllQuadrants[k]->x();
                LocalYForSecondCurve[countSecondCurve] = maximumPointsInAllQuadrants[k]->y();
                countSecondCurve++;
            }
        }

        for(int j = 0; j < pointsAboveCurve.size(); j++)
        {
            LocalXForSecondCurve[countSecondCurve + j] = pointsAboveCurve[j].x();
            LocalYForSecondCurve[countSecondCurve + j] = pointsAboveCurve[j].y();
        }

        //built second curve out of maximum points per quadrant and all points above the first curve
        polynomialfit(LocalXForSecondCurve, LocalYForSecondCurve, degree, info, curve, repo);
        //return(curve);

        //Also build a SISL-curve from this data for intersection calculation
        //using transformed global coordinates
        vector<Vec3> curvePoints;
        if (outerMinX<minimum_x)
        {
            osg::Vec3 pointRotated = osg::Vec3(outerMinX, barycentriccalc(curve,outerMinX),averageZ);
            osg::Vec3 point = inverseRotationMatrix * pointRotated;
            curvePoints.push_back(point);
        }
        for (int i=0; i!=numEdgeSectors; i++)
        {
            double x = (-minimum_x+maximum_x)/(numEdgeSectors-1)*i+minimum_x;
            double y = barycentriccalc(curve,x);
            double z = averageZ;
            osg::Vec3 pointRotated = osg::Vec3(x,y,z);
            osg::Vec3 point = inverseRotationMatrix * pointRotated;
            curvePoints.push_back(point);
        }
        if (outerMaxX>maximum_x)
        {
            osg::Vec3 pointRotated = osg::Vec3(outerMaxX, barycentriccalc(curve,outerMaxX),averageZ);
            osg::Vec3 point = inverseRotationMatrix * pointRotated;
            curvePoints.push_back(point);
        }

        int num_points = curvePoints.size();
        double *pointsSISLCurve = new double[2*curvePoints.size()];
        int *type= new int[curvePoints.size()];

        int i=0;
        for (auto it=curvePoints.begin(); it!= curvePoints.end(); it++)
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints curvePoints x: %f, y: %f, z: %f \n ",it->x(), it->y(), it->z() );
            pointsSISLCurve[i*2]=it->x();
            pointsSISLCurve[i*2+1]=it->y();
            type[i]=1;
            osg::Vec3 point = rotationMatrixToWorld * *it;
            highlightPoint(point, blue);
            i++;
        }
        const double cstartpar = 0;
            fprintf(stderr, "NurbsSurface::surfaceInfo::edgeByPoints trying sisl curve\n");
        try
        {
            double cendpar;

            double* gpar = 0;
            int jnbpar;
            int jstat;

            s1356(pointsSISLCurve,        // pointer to where the point coordinates are stored
                  num_points,    // number of points to be interpolated
                  2,             // the dimension
                  type,          // what type of information is stored at a particular point
                  0,             // no additional condition at start point
                  0,             // no additional condition at end point
                  1,             // open curve
                  3,             // order of the spline curve to be produced
                  cstartpar,     // parameter value to be used at start of curve
                  &cendpar,      // parameter value at the end of the curve (to be determined)
                  &result_curve, // the resulting spline curve (to be determined)
                  &gpar,         // pointer to the parameter values of the points in the curve
                  // (to be determined)
                  &jnbpar,       // number of unique parameter values (to be determined)
                  &jstat);       // status message

            if (jstat < 0) {
                throw runtime_error("Error occured inside call to SISL routine.");
            } else if (jstat > 0) {
                std::cerr << "WARNING: warning occured inside call to SISL routine. \n" << std::endl;
            }
            delete []pointsSISLCurve;
            delete[] type;
            resultCurveInfo.curve = result_curve;
            resultCurveInfo.startPar = cstartpar;
            resultCurveInfo.endPar = cendpar;
            fprintf(stderr,"start value %f end value %f\n",cstartpar,cendpar);

            // cleaning up
            //freeCurve(result_curve);
            free(gpar);
        }
        catch (exception& e)
        {
            std::cerr << "Exception thrown: " << e.what() << std::endl;
            return -1;
        }
    }
    return 0;
}

//method for creating edges in unsorted points
//
//    local_x -> local x-value for creating upper edge
//        for upper edge:   local_x = 0
//        for right edge:   local_x = 1
//    local_y -> local y-value for creating upper edge
//        for upper edge:   local_y = 1
//        for right edge:   local_y = 0
//    change -> changes upper edge to lower edge and right edge to left edge
//        without changing:   change = 1
//        with changing:      change = -1
int NurbsSurface::surfaceInfo::edge(vector<osg::Vec3> all_points, int local_x, int local_y, int change, curveInfo &resultCurveInfo)
{
    SISLCurve *result_curve = 0;

    numberOfAllPoints = all_points.size();                                                   //number of points
    maximum_x = -FLT_MAX;                                                                    //maximum local x-value
    minimum_x = FLT_MAX;                                                                     //minimum local x-value
    maximum_y = -FLT_MAX;                                                                    //maximum local y-value
    minimum_y = FLT_MAX;                                                                     //minimum local y-value

    double sumZ = 0.0;
    for (std::vector<osg::Vec3>::const_iterator it = all_points.begin(); it !=all_points.end(); it++)
    {
        sumZ += it->_v[2];
    }
    double averageZ = sumZ/all_points.size();

    for(int i = 0; i < numberOfAllPoints; i++)
    {                                             //search for minimum and maximum local x and y
        if ((all_points[i][local_x]) > maximum_x) {
            maximum_x = all_points[i][local_x];
        }
        if ((all_points[i][local_x]) < minimum_x) {
            minimum_x = all_points[i][local_x];
        }
        if ((change * all_points[i][local_y]) > maximum_y) {
            maximum_y = all_points[i][local_y];
        }
        if ((change * all_points[i][local_y]) < minimum_y) {
            minimum_y = all_points[i][local_y];
        }
    }
    //fprintf(stderr, "minimum x: %f y: %f, maximum x: %f y:%f\n", minimum_x, minimum_y, maximum_x, maximum_y);


    int numberOfQuadrants = numEdgeSectors;                                                  //number of quadrants
    float widthOfQuadrant = (maximum_x-minimum_x)/numberOfQuadrants;                         //width of quadrant


    real_1d_array LocalXForFirstCurve;                                                       //array for all local x-values with maximum value in their quadrant
    real_1d_array LocalYForFirstCurve;                                                       //array for all local y-values with maximum value in their quadrant
    real_1d_array LocalXForSecondCurve;                                                      //array for all local x-values witch are needed for curve interpolation
    real_1d_array LocalYForSecondCurve;                                                      //array for all local y-values witch are needed for curve interpolation


    std::vector<osg::Vec3*> maximumPointsInAllQuadrants;
    maximumPointsInAllQuadrants.resize(numberOfQuadrants, nullptr);
    std::vector<osg::Vec3> pointsAboveCurve;


    //initialize spline firstCurveWithMaximumPointsPerQuadrant
    ae_int_t degree = 3;

    ae_int_t info;
    barycentricinterpolant firstCurveWithMaximumPointsPerQuadrant;
    polynomialfitreport repo;
    //double test;

    //initialize spline curve
    barycentricinterpolant curve;

    int numberOfSectorsWithMaximum = 0;


    //find all maximum points in the quadrants
    if (minimum_x<maximum_x)
    {
        for(int i = 0; i < numberOfAllPoints; i++) {
            int j=int((all_points[i][local_x]-minimum_x)/(maximum_x-minimum_x)*numberOfQuadrants);
            j=min(numberOfQuadrants-1,j);

            //for(int j = 0; j < numberOfQuadrants; j++) {
            //if (all_points[i][local_x] >= (minimum_x + j*widthOfQuadrant) && all_points[i][local_x] < (minimum_x + (j+1)*widthOfQuadrant)) {
            if (!maximumPointsInAllQuadrants[j]) {
                maximumPointsInAllQuadrants[j] = &all_points[i];
                numberOfSectorsWithMaximum++;
            }
            else{
                if (change * all_points[i][local_y] > change * maximumPointsInAllQuadrants[j]->_v[local_y]) {
                    maximumPointsInAllQuadrants[j] = &all_points[i];
                }
                //}
            }
            if (maximumPointsInAllQuadrants[j])
            {
                //fprintf(stderr, "maximum point in quadrant %i is now x: %f y: %f\n",j,maximumPointsInAllQuadrants[j]->_v[0],maximumPointsInAllQuadrants[j]->_v[1]);
            }
        }
    }

    //only built a curve, when there are minimal two points
    if (numberOfSectorsWithMaximum > 1) {

        LocalXForFirstCurve.setlength(numberOfSectorsWithMaximum);
        LocalYForFirstCurve.setlength(numberOfSectorsWithMaximum);

        int countFirstCurve = 0;

        for(int k = 0; k < maximumPointsInAllQuadrants.size(); k++) {
            if (!maximumPointsInAllQuadrants[k]) {
            }
            else{
                LocalXForFirstCurve[countFirstCurve] = maximumPointsInAllQuadrants[k]->_v[local_x];
                LocalYForFirstCurve[countFirstCurve] = maximumPointsInAllQuadrants[k]->_v[local_y];
                countFirstCurve++;
            }
        }

        //built first curve out of maximum points per quadrant
        polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, firstCurveWithMaximumPointsPerQuadrant, repo);

        //compare all points with first curve, if they are above or below
        for(int i = 0; i < numberOfAllPoints; i++) {
            if (change * all_points[i][local_y] > change * barycentriccalc(firstCurveWithMaximumPointsPerQuadrant, all_points[i][local_x])) {
                pointsAboveCurve.push_back(all_points[i]);
            }
        }


        LocalXForSecondCurve.setlength(numberOfSectorsWithMaximum + pointsAboveCurve.size());
        LocalYForSecondCurve.setlength(numberOfSectorsWithMaximum + pointsAboveCurve.size());

        int countSecondCurve = 0;

        for(int k = 0; k < maximumPointsInAllQuadrants.size(); k++) {
            if (!maximumPointsInAllQuadrants[k]) {
            }
            else{
                LocalXForSecondCurve[countSecondCurve] = maximumPointsInAllQuadrants[k]->_v[local_x];
                LocalYForSecondCurve[countSecondCurve] = maximumPointsInAllQuadrants[k]->_v[local_y];
                countSecondCurve++;
            }
        }

        for(int j = 0; j < pointsAboveCurve.size(); j++) {
            LocalXForSecondCurve[countSecondCurve + j] = pointsAboveCurve[j][local_x];
            LocalYForSecondCurve[countSecondCurve + j] = pointsAboveCurve[j][local_y];
        }

        //built second curve out of maximum points per quadrant and all points above the first curve
        polynomialfit(LocalXForSecondCurve, LocalYForSecondCurve, degree, info, curve, repo);
        //return(curve);

        //Also build a SISL-curve from this data for intersection calculation
        //using transformed global coordinates
        int num_points = numberOfQuadrants;
        double *pointsSISLCurve = new double[2*num_points];
        int *type= new int[num_points];
        for (int i=0; i!=num_points; i++)
        {
            double x = (-minimum_x+maximum_x)/(num_points-1)*i+minimum_x;
            pointsSISLCurve[i*2+local_y]=barycentriccalc(curve,x);
            pointsSISLCurve[i*2+local_x]=x;
            type[i]=1;
            osg::Vec3 point = osg::Vec3(pointsSISLCurve[i*2],pointsSISLCurve[i*2+1],averageZ);
            point = rotationMatrixToWorld * point;
            highlightPoint(point, edgeColor);
            fprintf(stderr, "highlighting Point %f %f %f\n", point.x(), point.y(), point.z());
        }
        const double cstartpar = 0;
        try {

            double cendpar;

            double* gpar = 0;
            int jnbpar;
            int jstat;

            s1356(pointsSISLCurve,        // pointer to where the point coordinates are stored
                  num_points,    // number of points to be interpolated
                  2,             // the dimension
                  type,          // what type of information is stored at a particular point
                  0,             // no additional condition at start point
                  0,             // no additional condition at end point
                  1,             // open curve
                  3,             // order of the spline curve to be produced
                  cstartpar,     // parameter value to be used at start of curve
                  &cendpar,      // parameter value at the end of the curve (to be determined)
                  &result_curve, // the resulting spline curve (to be determined)
                  &gpar,         // pointer to the parameter values of the points in the curve
                  // (to be determined)
                  &jnbpar,       // number of unique parameter values (to be determined)
                  &jstat);       // status message

            if (jstat < 0) {
                throw runtime_error("Error occured inside call to SISL routine.");
            } else if (jstat > 0) {
                std::cerr << "WARNING: warning occured inside call to SISL routine. \n" << std::endl;
            }
            delete []pointsSISLCurve;
            delete[] type;
            resultCurveInfo.curve = result_curve;
            resultCurveInfo.startPar = cstartpar;
            resultCurveInfo.endPar = cendpar;
            fprintf(stderr,"start value %f end value %f\n",cstartpar,cendpar);

            // cleaning up
            //freeCurve(result_curve);
            free(gpar);
        } catch (exception& e) {
            std::cerr << "Exception thrown: " << e.what() << std::endl;
            return -1;
        }
    }

    //return empty curve if number of sectors with maximum is lower than 1
    return 0;
}

void NurbsSurface::surfaceInfo::createRBFModel()
{
    rbfcreate(2,1,model);
    rbfsetalgohierarchical(model, 1.0, 5, 0.0);
}

void NurbsSurface::surfaceInfo::updateModel()
{
    //updates receivedPointsRotated and xy
    fprintf(stderr, "NurbsSurface::updateModel() \n");
    //xy.setlength(receivedPoints.size(),3);

    centroid=Vec3(0.0,0.0,0.0);
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        centroid = centroid + *iter;
    }
    //Calculate centroid
    fprintf(stderr, "Calculate centroid \n");
    centroid = centroid / receivedPoints.size();

    fprintf(stderr,"p: %f %f %f",centroid.x(), centroid.y(), centroid.z());

    //Calculate normal of plane or let user pick
    
    //points relative to centroid
    //fprintf(stderr, "points relative to centroid \n");
    real_2d_array pointsRelativeToCentroid;
    pointsRelativeToCentroid.setlength(receivedPoints.size(), 3);
    int i=0;
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        pointsRelativeToCentroid[i][0]=iter->_v[0]-centroid._v[0];
        pointsRelativeToCentroid[i][1]=iter->_v[1]-centroid._v[1];
        pointsRelativeToCentroid[i][2]=iter->_v[2]-centroid._v[2];
        i++;
    }

    fprintf(stderr, "Covariance matrix \n");
    real_2d_array covarianceMatrix;
    //covarianceMatrix.setlength(3,3);
    covm(pointsRelativeToCentroid, covarianceMatrix);
    fprintf(stderr, "Covariance matrix has length: %li x %li \n", (long)covarianceMatrix.rows(), (long)covarianceMatrix.cols());
    fprintf(stderr,"covariance matrix: %f %f %f\n",covarianceMatrix[0][0], covarianceMatrix[0][1], covarianceMatrix[0][2]);
    fprintf(stderr,"covariance matrix: %f %f %f\n",covarianceMatrix[1][0], covarianceMatrix[1][1], covarianceMatrix[1][2]);
    fprintf(stderr,"covariance matrix: %f %f %f\n",covarianceMatrix[2][0], covarianceMatrix[2][1], covarianceMatrix[2][2]);

    //Calculate eigenvector
    fprintf(stderr, "Calculate eigenvector \n");
    eigsubspacestate state;
    eigsubspacecreate(3,3, state);
    real_1d_array eigenvalue;
    real_2d_array eigenvector;
    eigsubspacereport eigenRep;
    eigsubspacesolvedenses(state,covarianceMatrix, true, eigenvalue, eigenvector, eigenRep);
    fprintf(stderr,"eigenvector 0: %f %f %f\n",eigenvector[0][0], eigenvector[1][0], eigenvector[2][0]);
    fprintf(stderr,"eigenvector 1: %f %f %f\n",eigenvector[0][1], eigenvector[1][1], eigenvector[2][1]);
    fprintf(stderr,"eigenvector 2: %f %f %f\n",eigenvector[0][2], eigenvector[1][2], eigenvector[2][2]);
    
    fprintf(stderr,"eigenvalues: %f %f %f\n",eigenvalue[0], eigenvalue[1], eigenvalue[2]);
    //calculate coordinate axis and rotation matrix
    osg::Vec3 zlocal = osg::Vec3(eigenvector[0][2], eigenvector[1][2], eigenvector[2][2]);
    zlocal.normalize();
    
    osg::Vec3 ylocal;
    double maxZAxisAbsValue=fmax(fmax(fabs(eigenvector[0][2]),fabs(eigenvector[1][2])), fabs(eigenvector[2][2]));
    if (maxZAxisAbsValue == fabs(eigenvector[0][2]))
        ylocal = osg::Vec3(0.0, - eigenvector[2][2], eigenvector[1][2]);
    if (maxZAxisAbsValue == fabs(eigenvector[1][2]))
        ylocal = osg::Vec3(-eigenvector[2][2], 0.0, eigenvector[0][2]);
    if (maxZAxisAbsValue == fabs(eigenvector[2][2]))
        ylocal = osg::Vec3(-eigenvector[1][2], eigenvector[0][2], 0.0);
    ylocal.normalize();
    osg::Vec3 xlocal = ylocal^zlocal;
    xlocal.normalize();
    fprintf(stderr,"xlocal: %f %f %f\n",xlocal[0], xlocal[1], xlocal[2]);
    fprintf(stderr,"ylocal: %f %f %f\n",ylocal[0], ylocal[1], ylocal[2]);
    fprintf(stderr,"zlocal: %f %f %f\n",zlocal[0], zlocal[1], zlocal[2]);
    
    rotationMatrixToWorld = osg::Matrixd(xlocal[0],ylocal[0],zlocal[0],0.0,xlocal[1],ylocal[1],zlocal[1],0.0,xlocal[2],ylocal[2],zlocal[2],0.0,0.0,0.0,0.0,1.0);
    rotationMatrixToLocal.invert(rotationMatrixToWorld);
    osg::Vec3 testVec = osg::Vec3(1.0,0.0,0.0);
    osg::Vec3 testVecRotated = rotationMatrixToLocal * testVec;

    fprintf(stderr,"testVecRotated: %f %f %f\n",testVecRotated[0], testVecRotated[1], testVecRotated[2]);
    //Rotate points
    receivedPointsRotated.clear();
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        osg::Vec3 rotatedPoint = rotationMatrixToLocal * *iter;
        receivedPointsRotated.push_back(rotatedPoint);
        i++;
    }

    receivedBoundaryPointsRotated.clear();
    for (std::vector<osg::Vec3>::const_iterator iter = receivedBoundaryPoints.begin() ; iter != receivedBoundaryPoints.end(); iter++)
    {
        osg::Vec3 rotatedPoint = rotationMatrixToLocal * *iter;
        receivedBoundaryPointsRotated.push_back(rotatedPoint);
        i++;
    }

    centroidRotated = rotationMatrixToLocal * centroid;

    i=0;
    xy.setlength(receivedPointsRotated.size(),3);
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPointsRotated.begin() ; iter != receivedPointsRotated.end(); iter++)
    {
        //Create real_2d_array holding points
        xy[i][0]=iter->_v[0];
        xy[i][1]=iter->_v[1];
        xy[i][2]=iter->_v[2];
        i++;
    }
    //fprintf(stderr,"xy updated\n");

    //Create RBFModel
    rbfsetpoints(model,xy);
    //fprintf(stderr,"xy points set\n");

    rbfreport rep;
    rbfbuildmodel(model, rep);
    //fprintf(stderr,"Model updated\n");
}

void
NurbsSurface::surfaceInfo::highlightPoint(osg::Vec3& newSelectedPoint, osg::Vec4 colour)
{

    osg::Matrix *sphereTransformationMatrix = new osg::Matrix;
    sphereTransformationMatrix->makeTranslate(newSelectedPoint);

    osg::MatrixTransform *sphereTransformation = new osg::MatrixTransform;
    sphereTransformation->setMatrix(*sphereTransformationMatrix);

    transformMatrices.push_back(sphereTransformation);

    osg::Geode *sphereGeode = new osg::Geode;
    sphereTransformation->addChild(sphereGeode);
    osg::Sphere *selectedSphere = new osg::Sphere(Vec3(.0f,0.f,0.f),1.0f);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::StateSet* stateSet = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    osg::ShapeDrawable *selectedSphereDrawable = new osg::ShapeDrawable(selectedSphere, hint);
    osg::Material *selMaterial = new osg::Material();
    sphereGeode->addDrawable(selectedSphereDrawable);

    splinePointsGroup->addChild(sphereTransformation);
    selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, colour);
    selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, colour);
    selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
    selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
    selMaterial->setColorMode(osg::Material::OFF);

    stateSet->setAttribute(selMaterial);
    selectedSphereDrawable->setStateSet(stateSet);
}


void NurbsSurface::surfaceInfo::resize()
{
    osg::Vec3 wpoint1 = osg::Vec3(0, 0, 0);
    osg::Vec3 wpoint2 = osg::Vec3(0, 0, 1);
    osg::Vec3 opoint1 = wpoint1 * cover->getInvBaseMat();
    osg::Vec3 opoint2 = wpoint2 * cover->getInvBaseMat();

    //distance formula
    osg::Vec3 wDiff = wpoint2 - wpoint1;
    osg::Vec3 oDiff = opoint2 - opoint1;
    double distWld = wDiff.length();
    double distObj = oDiff.length();

    //controls the sphere size
    double scaleFactor = sphereSize * distObj / distWld;
    //scaleFactor = 1.1f;

    // scale all selected points
    for (std::vector<osg::MatrixTransform*>::iterator iter = transformMatrices.begin(); iter !=transformMatrices.end(); iter++)
    {
        osg::Matrix sphereMatrix = (*iter)->getMatrix();
        Vec3 translation = sphereMatrix.getTrans();
        sphereMatrix.makeScale(scaleFactor, scaleFactor, scaleFactor);
        sphereMatrix.setTrans(translation);
        (*iter)->setMatrix(sphereMatrix);
    }
}

void NurbsSurface::updateMessage()
{
    //send message to PointCloudPlugin
    cover->sendMessage(NULL, "PointCloud", PluginMessageTypes::PointCloudSurfaceMsg, sizeof(surfaceGeodes), &surfaceGeodes);
}

void NurbsSurface::selectionSetMessage()
{
    //Tell PointCloud-plugin which surface is currently selected
    cover->sendMessage(NULL, "PointCloud", PluginMessageTypes::PointCloudSelectionSetMsg, sizeof(currentSurface->surfaceIndex), &(currentSurface->surfaceIndex));
}

void NurbsSurface::selectionIsBoundaryMessage()
{
    // selected points mark the boundary
    cover->sendMessage(NULL, "PointCloud", PluginMessageTypes::PointCloudSelectionIsBoundaryMsg, sizeof(m_selectionIsBoundary), &(m_selectionIsBoundary));
}

void NurbsSurface::createNewSurface()
{
    surfaces.push_back(surfaceInfo());
    if (surfaceSelectionSlider)
        surfaceSelectionSlider->setBounds(0,surfaces.size()-1);
    currentSurface = &surfaces.back();
    currentSurface->surfaceIndex = surfaces.size()-1;
    currentSurface->splinePointsGroup = new osg::Group();
    currentSurface->createRBFModel();
    cover->getObjectsRoot()->addChild(currentSurface->splinePointsGroup.get());
    selectionSetMessage();
    if (surfaceSelectionSlider)
        surfaceSelectionSlider->setValue(currentSurface->surfaceIndex);
}

void NurbsSurface::updateUI()
{
    char currentSurfaceName[100];
    sprintf(currentSurfaceName,"Current Surface: %d",currentSurface->surfaceIndex);
    currentSurfaceLabel->setText(currentSurfaceName);
    surfaceSelectionSlider->setBounds(0,surfaces.size()-1);
}

void NurbsSurface::setSelectionIsBoundary(bool selectionIsBoundary)
{
    m_selectionIsBoundary = selectionIsBoundary;
    selectionIsBoundaryMessage();
}

COVERPLUGIN(NurbsSurface)
