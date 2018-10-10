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
    }
    );

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

NurbsSurface::NurbsSurface() : ui::Owner("NurbsSurface", cover->ui)
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
        for (vector<pointSelection>::const_iterator iter=selectedPoints->begin(); iter!=selectedPoints->end(); iter++)
        {
        Vec3 newSelectedPoint = Vec3(iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].x,
                                     iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].y,
                                     iter->file->pointSet[iter->pointSetIndex].points[iter->pointIndex].z);
        currentSurface->receivedPoints.push_back(newSelectedPoint);
        }
        fprintf(stderr, "Points received %zi\n", currentSurface->receivedPoints.size());
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
        calcEdges();
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
}

void NurbsSurface::surfaceInfo::evaluateCurveAtParam(curveInfo& curve, double paramFactor, vector<double> &point)
{
    int jstat; // status variable
    int temp;
    s1227(curve.curve,0,curve.startPar+paramFactor*(curve.endPar-curve.startPar),&temp,&point[0],&jstat);
}

vector<double> NurbsSurface::surfaceInfo::evaluateCurveAtParam(curveInfo& curve, double paramFactor)
{
    vector<double> result;
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


int NurbsSurface::surfaceInfo::edgeByPoints(std::vector<Vec3> all_points, Vec3 pointBegin, Vec3 pointEnd, curveInfo &resultCurveInfo)
{
    SISLCurve *result_curve = 0;
    //int numPoints = all_points.size();

    //rotation of points
    Vec3 edgeDirection = pointEnd - pointBegin;
    edgeDirection.normalize();
    Matrixd rotationMatrix;
    Matrixd inverseRotationMatrix;
    Vec3 unitAxis = Vec3(1.0, 0.0, 0.0);
    rotationMatrix.makeRotate(edgeDirection,unitAxis);
    inverseRotationMatrix.inverse(rotationMatrix);
    for (auto it=all_points.begin(); it!=all_points.end(); it++)
    {
        it->set(rotationMatrix * *it);
    }
    Vec3 pointBeginLocal = rotationMatrix * pointBegin;
    Vec3 pointEndLocal = rotationMatrix * pointEnd;

    //
    double sumZ = 0.0;
    for (std::vector<osg::Vec3>::const_iterator it = all_points.begin(); it !=all_points.end(); it++)
    {
        sumZ += it->_v[2];
    }
    double averageZ = sumZ/all_points.size();

    float minimum_x = pointBeginLocal.x();
    float maximum_x = pointEndLocal.x();

    std::vector<osg::Vec3*> maximumPointsInAllQuadrants;
    maximumPointsInAllQuadrants.resize(numEdgeSectors, nullptr);
    real_1d_array LocalXForFirstCurve;
    real_1d_array LocalYForFirstCurve;

    int numberOfSectorsWithMaximum = 0;
    for(auto it=all_points.begin(); it != all_points.end(); it++)
    {
        int j=int((it->x()-minimum_x)/(maximum_x-minimum_x)*numEdgeSectors);
        j=min(numEdgeSectors-1,j);
        if (!maximumPointsInAllQuadrants[j])
        {
            maximumPointsInAllQuadrants[j] = &*it;
            numberOfSectorsWithMaximum++;
        }
        else
        {
            if (it->y() > maximumPointsInAllQuadrants[j]->y())
            {
                maximumPointsInAllQuadrants[j] = &*it;
            }
        }
    }
    //initialize spline firstCurveWithMaximumPointsPerQuadrant
    ae_int_t degree = 3;

    ae_int_t info;
    barycentricinterpolant firstCurveWithMaximumPointsPerQuadrant;
    polynomialfitreport repo;

    //initialize spline curve
    barycentricinterpolant curve;

    if (numberOfSectorsWithMaximum > 1)
    {
        LocalXForFirstCurve.setlength(numberOfSectorsWithMaximum);
        LocalYForFirstCurve.setlength(numberOfSectorsWithMaximum);

        int countFirstCurve = 0;

        for(std::vector<Vec3*>::iterator it = maximumPointsInAllQuadrants.begin(); it != maximumPointsInAllQuadrants.end(); it++)
        {
            LocalXForFirstCurve[countFirstCurve] = (*it)->x();
            LocalYForFirstCurve[countFirstCurve] = (*it)->y();
            countFirstCurve++;
        }

        //built first curve out of maximum points per quadrant
        polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, firstCurveWithMaximumPointsPerQuadrant, repo);

        std::vector<osg::Vec3> pointsAboveCurve;

        //compare all points with first curve, if they are above or below
        for(auto it = all_points.begin(); it != all_points.end(); it++)
        {
            if (it->y() > barycentriccalc(firstCurveWithMaximumPointsPerQuadrant, it->x()))
            {
                pointsAboveCurve.push_back(*it);
            }
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
        int num_points = numEdgeSectors;
        double *pointsSISLCurve = new double[2*num_points];
        int *type= new int[num_points];
        for (int i=0; i!=num_points; i++)
        {
            double x = (-minimum_x+maximum_x)/(num_points-1)*i+minimum_x;
            pointsSISLCurve[i*2+1]=barycentriccalc(curve,x);
            pointsSISLCurve[i*2]=x;
            type[i]=1;
            osg::Vec3 point = osg::Vec3(pointsSISLCurve[i*2],pointsSISLCurve[i*2+1],averageZ);
            point = rotationMatrixToWorld * point;
            highlightPoint(point);
            //fprintf(stderr, "highlighting Point %f %f %f\n", point.x(), point.y(), point.z());
        }
        const double cstartpar = 0;
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

void NurbsSurface::surfaceInfo::createRBFModel()
{
    rbfcreate(2,1,model);
    rbfsetalgohierarchical(model, 1.0, 5, 0.0);
}

void NurbsSurface::surfaceInfo::updateModel()
{
    fprintf(stderr, "NurbsSurface::updateModel() \n");
    //xy.setlength(receivedPoints.size(),3);
    int i=0;
    osg::Vec3 centroid = osg::Vec3(0.0, 0.0, 0.0); 
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        //Create real_2d_array holding points
        //xy[i][0]=iter->_v[0];
        //xy[i][1]=iter->_v[1];
        //xy[i][2]=iter->_v[2];
        centroid = centroid + *iter;
        i++;
    }
    //Calculate centroid
    fprintf(stderr, "Calculate centroid \n");
    centroid = centroid / receivedPoints.size();

    fprintf(stderr,"p: %f %f %f",centroid.x(), centroid.y(), centroid.z());

    //Calculate normal of plane or let user pick
    
    //points relative to centroid
    //fprintf(stderr, "points relative to centroid \n");
    real_2d_array pointsRelativeToCentroid;
    //pointsRelativeToCentroid.setlength(3, receivedPoints.size());
    pointsRelativeToCentroid.setlength(receivedPoints.size(), 3);
    i=0;
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPoints.begin() ; iter != receivedPoints.end(); iter++)
    {
        //fprintf(stderr, "points relative to centroid %i \n",i);
        //Create real_2d_array holding points relative to centroid
        /*pointsRelativeToCentroid[0][i]=iter->_v[0]-centroid._v[0];
        pointsRelativeToCentroid[1][i]=iter->_v[1]-centroid._v[1];
        pointsRelativeToCentroid[2][i]=iter->_v[2]-centroid._v[2];
        fprintf(stderr,"point %i: %f %f %f \n",i, pointsRelativeToCentroid[0][i], pointsRelativeToCentroid[1][i], pointsRelativeToCentroid[2][i]);*/
        pointsRelativeToCentroid[i][0]=iter->_v[0]-centroid._v[0];
        pointsRelativeToCentroid[i][1]=iter->_v[1]-centroid._v[1];
        pointsRelativeToCentroid[i][2]=iter->_v[2]-centroid._v[2];
        fprintf(stderr,"point %i: %f %f %f \n",i, pointsRelativeToCentroid[i][0], pointsRelativeToCentroid[i][1], pointsRelativeToCentroid[i][2]);
        //fprintf(stderr,"xy %i: %f %f %f \n",i, xy[i][0], xy[i][1], xy[i][2]);
        i++;
    }

    fprintf(stderr, "Covariance matrix \n");
    real_2d_array covarianceMatrix;
    //covarianceMatrix.setlength(3,3);
    covm(pointsRelativeToCentroid, covarianceMatrix);
    fprintf(stderr, "Covariance matrix has length: %i x %i \n", covarianceMatrix.rows(), covarianceMatrix.cols());
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
    fprintf(stderr, "eigenvector has length: %i x %i \n",  eigenvector.rows(), eigenvector.cols());
    //fprintf(stderr,"Eigenvector: %f %f %f \n",eigenvector[0], eigenvector[1],eigenvector[0+2*eigenvector.getstride()]);
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
        fprintf(stderr,"rotatedPoint: %f %f %f\n",rotatedPoint[0], rotatedPoint[1], rotatedPoint[2]);
        receivedPointsRotated.push_back(rotatedPoint);
        i++;
    }
    i=0;
    xy.setlength(receivedPointsRotated.size(),3);
    for (std::vector<osg::Vec3>::const_iterator iter = receivedPointsRotated.begin() ; iter != receivedPointsRotated.end(); iter++)
    {
        fprintf(stderr,"i %i\n",i);
        //Create real_2d_array holding points
        xy[i][0]=iter->_v[0];
        xy[i][1]=iter->_v[1];
        xy[i][2]=iter->_v[2];

        fprintf(stderr, "rotatedPointxy: %f %f %f \n", xy[i][0],xy[i][1],xy[i][2]);
        i++;
    }
    fprintf(stderr,"xy updated\n");

    //Create RBFModel
    rbfsetpoints(model,xy);
    fprintf(stderr,"xy points set\n");

    rbfreport rep;
    rbfbuildmodel(model, rep);
    fprintf(stderr,"Model updated\n");
}

void
NurbsSurface::surfaceInfo::highlightPoint(osg::Vec3& newSelectedPoint)
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

void NurbsSurface::createNewSurface()
{
    surfaces.push_back(surfaceInfo());
    currentSurface = &surfaces.back();
    currentSurface->surfaceIndex = surfaces.size()-1;
    currentSurface->splinePointsGroup = new osg::Group();
    currentSurface->createRBFModel();
    cover->getObjectsRoot()->addChild(currentSurface->splinePointsGroup.get());
}

void NurbsSurface::updateUI()
{
    char currentSurfaceName[100];
    sprintf(currentSurfaceName,"Current Surface: %d",currentSurface->surfaceIndex);
    currentSurfaceLabel->setText(currentSurfaceName);
    surfaceSelectionSlider->setBounds(0,surfaces.size()-1);
}

COVERPLUGIN(NurbsSurface)
