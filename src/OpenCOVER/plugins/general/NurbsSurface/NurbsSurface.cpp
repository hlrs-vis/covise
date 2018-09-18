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
#define QT_CLEAN_NAMESPACE
#include <QString>
#include <QDebug>
#include "qregexp.h"
#include "qdir.h"

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
//#include <cover/ui/Button.h>

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
    saveButton_ = new ui::Action(NurbsSurfaceMenu, "SaveFile");
    saveButton_->setText("SaveFile");
    saveButton_->setCallback([this](){
        saveFile("test.obj");
    });

    orderUSlider = new ui::Slider(NurbsSurfaceMenu, "order_U");
    orderVSlider = new ui::Slider(NurbsSurfaceMenu, "order_V");

    orderUSlider->setIntegral(true);
    orderUSlider->setText("Order U");
    orderUSlider->setBounds(2, 4);
    orderUSlider->setValue(order_U);;
    orderUSlider->setCallback([this](int val, bool released)
    {
        destroy();
        order_U=val;
        updateSurface();
    }
    );



    orderVSlider->setIntegral(true);
    orderVSlider->setText("Order V");
    orderVSlider->setBounds(2, 4);
    orderVSlider->setValue(order_V);;
    orderVSlider->setCallback([this](int val, bool released)
    {
        destroy();
        order_V=val;
        updateSurface();
    }
    );

    numEdgeSectorsSlider = new ui::Slider(NurbsSurfaceMenu, "edge_sectors");
    numEdgeSectorsSlider->setIntegral(true);
    numEdgeSectorsSlider->setText("Number of sectors for edge interpolation");
    numEdgeSectorsSlider->setBounds(2,10);
    numEdgeSectorsSlider->setValue(numEdgeSectors);
    numEdgeSectorsSlider->setCallback([this](int val, bool released)
    {
        destroy();
        numEdgeSectors=val;
        updateSurface();
    }
    );

}

void NurbsSurface::saveFile(const std::string &fileName)
{
        osgDB::writeNodeFile(*geode, fileName.c_str());
}

bool NurbsSurface::init()
{
        if (cover->debugLevel(3))
                fprintf(stderr, "\n--- NurbsSurface::init\n");

        initUI();
        createRBFModel();
        return true;
}

void NurbsSurface::computeSurface(double* points)
{
    // generating interpolating surface
    for (int i = 0; i < num_surf; ++i) {

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
                vertices->push_back(osg::Vec3(result_surf->ecoef[p+0], result_surf->ecoef[p+1], result_surf->ecoef[p+2]));
                normals->push_back(osg::Vec3(normal[p+0], normal[p+1], normal[p+2]));
                colors->push_back(_color);
                p+=3;
                vertices->push_back(osg::Vec3(result_surf->ecoef[p+0], result_surf->ecoef[p+1], result_surf->ecoef[p+2]));
                normals->push_back(osg::Vec3(normal[p+0], normal[p+1], normal[p+2]));
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

        // add the points geomtry to the geode.
        cover->getObjectsRoot()->addChild(geode.get());
        surfaces.push_back(geode.get());

        // send back surface to PointCloudPlugin for intersection testing and pointcloud manipulation


        freeSurf(result_surf);
    }
}

NurbsSurface::NurbsSurface() : ui::Owner("NurbsSurface", cover->ui)
{
    splinePointsGroup = new osg::Group();
    cover->getObjectsRoot()->addChild(splinePointsGroup.get());
    updateSurface();

}

bool NurbsSurface::destroy()
{
    cover->getObjectsRoot()->removeChild(geode.get());
    return true;
}


// this is called if the plugin is removed at runtime
NurbsSurface::~NurbsSurface()
{
    fprintf(stderr, "Goodbye\n");
    cover->getObjectsRoot()->removeChild(splinePointsGroup.get());
}

void NurbsSurface::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::NurbsSurfacePointMsg)
    {
        vector<pointSelection> *selectedPoints = (vector<pointSelection> *)buf;
        receivedPoints.clear();
        for (vector<pointSelection>::const_iterator iter=selectedPoints->begin(); iter!=selectedPoints->end(); iter++)
        {
        Vec3 newSelectedPoint = Vec3(iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].x,
                                     iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].y,
                                     iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].z);
        receivedPoints.push_back(newSelectedPoint);
        }
                fprintf(stderr, "Points received %zi\n", receivedPoints.size());
                updateSurface();
    }
}

void NurbsSurface::updateSurface()
{
    while (splinePointsGroup->getNumChildren()>0)
    {
        splinePointsGroup->removeChildren(0,1);
    }
    transformMatrices.clear();
    if (receivedPoints.size()>20)
    {
        calcEdges();
        updateModel();
        resize();
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
        //pointTempLeftOne=evaluateCurveAtParam(left,1.0);
        //pointTempRightOne=evaluateCurveAtParam(right,1.0);
        evaluateCurveAtParam(left,0.0,pointTempLeftZero);
        evaluateCurveAtParam(right,0.0,pointTempRightZero);
        int jstat; // status variable
        int temp;
        int k=0;
        for (int i=0; i!=num_points_u; i++)
        {
            double paramFactor0=1.0/(num_points_u-1)*i;
            fprintf(stderr,"paramFactor0: %f ",paramFactor0);
            for (int j=0; j!=num_points_v;j++)
            {
                double paramFactor1=1.0/(num_points_v-1)*j;
                fprintf(stderr,"paramFactor1: %f \n",paramFactor1);
                evaluateCurveAtParam(upper,paramFactor0,pointTempUpper);
                evaluateCurveAtParam(lower,paramFactor0,pointTempLower);
                vector<double> pointTempUpperLower = pointTempUpper+(pointTempLower-pointTempUpper)*paramFactor1;
                evaluateCurveAtParam(left,paramFactor1,pointTempLeft);
                evaluateCurveAtParam(right,paramFactor1,pointTempRight);
                vector<double> pointTempLeftRight = (paramFactor0*(pointTempRight -(pointTempRightZero+(pointTempRightOne-pointTempRightZero)*paramFactor1))+
                                                     (1.0-paramFactor0)*(pointTempLeft -(pointTempLeftZero+(pointTempLeftOne-pointTempLeftZero)*paramFactor1)));
                interpolatedPoint = pointTempUpperLower + pointTempLeftRight;
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

        int l=0;
        while (l!=pointsNewSize)
        {
            fprintf(stderr,"p: %i %f ",l,pointsNew[l]);
            if (l % 3 ==2)
                fprintf(stderr,"\n");
            l++;
        }
        destroy();
        computeSurface(pointsNew);
		delete[] pointsNew;
    }
}

void NurbsSurface::evaluateCurveAtParam(curveInfo& curve, double paramFactor, vector<double> &point)
{
    int jstat; // status variable
    int temp;
    s1227(curve.curve,0,curve.startPar+paramFactor*(curve.endPar-curve.startPar),&temp,&point[0],&jstat);
}

vector<double> NurbsSurface::evaluateCurveAtParam(curveInfo& curve, double paramFactor)
{
    vector<double> result;
    int jstat; // status variable
    int temp;
    s1227(curve.curve,0,curve.startPar+paramFactor*(curve.endPar-curve.startPar),&temp,&result[0],&jstat);
    return result;
}

bool NurbsSurface::calcEdges()
{
    edge(receivedPoints,0,1,1,upper);
    edge(receivedPoints,1,0,1,right);
    edge(receivedPoints,0,1,-1,lower);
    edge(receivedPoints,1,0,-1,left);
    if (upper.curve && lower.curve && left.curve && right.curve)
    {
        curveCurveIntersection(upper.curve, upper.endPar, right.curve, right.endPar);
        curveCurveIntersection(lower.curve, lower.endPar, right.curve, right.startPar);
        curveCurveIntersection(lower.curve, lower.startPar, left.curve, left.startPar);
        curveCurveIntersection(upper.curve, upper.startPar, left.curve, left.endPar);
        /*fprintf(stderr, "upper curve end parameter %f \n", upper.endPar);
    fprintf(stderr, "upper curve start parameter %f \n", upper.startPar);
    fprintf(stderr, "right curve end parameter %f \n", right.endPar);
    fprintf(stderr, "right curve start parameter %f \n", right.startPar);*/
        return true;
    }
    return false;
}

bool NurbsSurface::curveCurveIntersection(SISLCurve *c1, double& c1Param, SISLCurve *c2, double& c2Param)
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

int NurbsSurface::edge(vector<osg::Vec3> all_points, int local_x, int local_y, int change, curveInfo &resultCurveInfo) {

    SISLCurve *result_curve = 0;

    numberOfAllPoints = all_points.size();                                                   //number of points
    maximum_x = -FLT_MAX;                                                                    //maximum local x-value
    minimum_x = FLT_MAX;                                                                     //minimum local x-value
    maximum_y = -FLT_MAX;                                                                    //maximum local y-value
    minimum_y = FLT_MAX;                                                                     //minimum local y-value



    for(int i = 0; i < numberOfAllPoints; i++) {                                             //search for minimum and maximum local x and y
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
            osg::Vec3 point = osg::Vec3(pointsSISLCurve[i*2],pointsSISLCurve[i*2+1],0.0);
            highlightPoint(point);
            //fprintf(stderr, "highlighting Point %f %f %f\n", point.x(), point.y(), point.z());
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

void NurbsSurface::createRBFModel()
{
    rbfcreate(2,1,model);
    rbfsetalgohierarchical(model, 1.0, 5, 0.0);
}

void NurbsSurface::updateModel()
{
    xy.setlength(receivedPoints.size(),3);
    int i=0;
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        xy[i][0]=iter->_v[0];
        xy[i][1]=iter->_v[1];
        xy[i][2]=iter->_v[2];
        i++;
    }
    rbfsetpoints(model,xy);
    rbfreport rep;
    rbfbuildmodel(model, rep);
}

void
NurbsSurface::highlightPoint(osg::Vec3& newSelectedPoint)
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
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.0, 0.6, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.0, 0.0, 0.6, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);

    stateSet->setAttribute(selMaterial);
    selectedSphereDrawable->setStateSet(stateSet);
}


void NurbsSurface::resize()
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
    cover->sendMessage(NULL, "PointCloud", PluginMessageTypes::PointCloudSurfaceMsg, sizeof(surfaces), &surfaces);
}



COVERPLUGIN(NurbsSurface)
