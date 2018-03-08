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
        computeSurface();        
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
        computeSurface();        
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
        return true;
}

void NurbsSurface::computeSurface()
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
                    cerr << "WARNING: warning occured inside call to SISL routine s1537. \n"
                    << endl;
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


            freeSurf(result_surf);


        }
}




NurbsSurface::NurbsSurface() : ui::Owner("NurbsSurface", cover->ui)
{

    computeSurface();

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
}

void NurbsSurface::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::NurbsSurfacePointMsg)
    {
        osg::Vec3 *selectedPoint = (osg::Vec3 *)buf;
        fprintf(stderr, "Point received %f %f %f", selectedPoint->x(), selectedPoint->y(), selectedPoint->z());
    }
}

alglib::barycentricinterpolant NurbsSurface::edge(vector<osg::Vec3> all_points, int local_x, int local_y, int change) {


    numberOfAllPoints = all_points.size();                                                   //number of points
    maximum_x = -FLT_MAX;                                                                    //maximum x-value
    minimum_x = FLT_MAX;                                                                     //minimum x-value
    maximum_y = -FLT_MAX;                                                                    //maximum y-value
    minimum_y = FLT_MAX;                                                                     //minimum y-value



    for(int i = 0; i < numberOfAllPoints; i++) {
        if ((change * all_points[i][local_x]) > maximum_x) {  //all_points[i]._v[i]
            maximum_x = all_points[i][local_x];
        }
        if ((change * all_points[i][local_x]) < minimum_x) {
            minimum_x = all_points[i][local_x];
        }
        if ((change * all_points[i][local_y]) > maximum_y) {
            maximum_y = all_points[i][local_y];
        }
        if ((change * all_points[i][local_y]) < minimum_y) {
            minimum_y = all_points[i][local_y];
        }
    }
    //fprintf(stderr,"%i %i %i\n", numberOfAllPoints, maximum_x, maximum_y);

    int numberOfQuadrants = 3;                                                               //number of quadrants
    float widthOfQuadrant = (maximum_x-minimum_x)/numberOfQuadrants;                         //width of quadrant


    std::vector< std::vector<float> > upperEdge;
    std::vector< std::vector<float> > upperEdgeTemp;
    real_1d_array LocalXForFirstCurve;
    real_1d_array LocalYForFirstCurve;


    std::vector<osg::Vec3*> maximumPointsInAllQuadrants;
    maximumPointsInAllQuadrants.resize(numberOfQuadrants, nullptr);
    std::vector<osg::Vec3> pointsAboveCurve;


    //initialize spline
    ae_int_t degree = 3;
    double spot = 2;
    ae_int_t info;
    barycentricinterpolant curve;
    polynomialfitreport repo;
    double test;

    int numberOfSectorsWithMaximum = 0;

    for(int i = 0; i < numberOfAllPoints; i++) {
        for(int j = 0; j < numberOfQuadrants; j++) {
            if (all_points[i][local_x] >= (minimum_x + j*widthOfQuadrant) && all_points[i][local_x] < (minimum_x + (j+1)*widthOfQuadrant)) {
                if (!maximumPointsInAllQuadrants[j]) {
                    maximumPointsInAllQuadrants[j] = &all_points[i];
                    numberOfSectorsWithMaximum++;
                }
                else{
                    if (all_points[i][local_y] > maximumPointsInAllQuadrants[j]->_v[local_y]) {
                        maximumPointsInAllQuadrants[j] = &all_points[i];
                    }
                }
            }
        }
    }

    if (numberOfSectorsWithMaximum > 1) {

        LocalXForFirstCurve.setlength(numberOfSectorsWithMaximum);
        LocalYForFirstCurve.setlength(numberOfSectorsWithMaximum);

        for(int k = 0; k < maximumPointsInAllQuadrants.size(); k++) {

            LocalXForFirstCurve[k] = maximumPointsInAllQuadrants[k]->_v[local_x];

            LocalYForFirstCurve[k] = maximumPointsInAllQuadrants[k]->_v[local_y];

        }

        polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, curve, repo);

        for(int i = 0; i < numberOfAllPoints; i++) {
            if (all_points[i][local_y] > barycentriccalc(curve, all_points[i][local_x])) {
                pointsAboveCurve.push_back(all_points[i]);
            }
        }
    }


    /*if (all_points[i][local_x] <= (minimum_x + widthOfQuadrant) && all_points[i][local_x] >= minimum_x) {                      //first quadrant
      if (all_points[i][local_y] > ) {

      }
      else {
      if (all_points[i][local_y] >  ) {

      }
      }

      }

      if (all_points[i][local_x] <= (minimum_x + 2*widthOfQuadrant) && all_points[i][local_x] > (minimum_x + widthOfQuadrant)) {      //second Quadrant

      }

      if (all_points[i][local_x] <= maximum_x && all_points[i][local_x] > (minimum_x + 2*widthOfQuadrant)) {                     //third Quadrant
      */


    /*
       if (upperEdgeTemp.size()<n)                                                         //Arrays bauen bis n erreicht
       {
       std::vector<float> r2(2);
       r2[0] = xy[i][x];
       r2[1] = xy[i][y];
       upperEdgeTemp.push_back(r2);

       LocalXForFirstCurve.setlength(upperEdgeTemp.size());
       LocalYForFirstCurve.setlength(upperEdgeTemp.size());

       for (int j=0; j<upperEdgeTemp.size();j++)                                       //Baue real_1d_array LocalXForFirstCurve
       {
       LocalXForFirstCurve[j] = upperEdgeTemp[j][x];
    //fprintf(stderr,"%i %f\n", upperEdgeTemp.size(), LocalXForFirstCurve[j]);
    }
    for (int j=0; j<upperEdgeTemp.size();j++)                                       //Baue real_1d_array LocalYForFirstCurve
    {
    LocalYForFirstCurve[j] = upperEdgeTemp[j][y];
    //fprintf(stderr,"%f\n", LocalYForFirstCurve[j]);
    }

    }*/
    /*
       else
       {
    //if-Schleife: liegt der Punkt außerhalb des Randes?
    if ((wechsel * xy[i][y]) > barycentriccalc(curve, xy[i][x]))
    {
    //Wenn Punkt hinzugefügt, Spline neu bauen, schlechtesten Punkt löschen
    std::vector<float> r3(2);
    r3[0] = xy[i][x];
    r3[1] = xy[i][y];
    upperEdgeTemp.push_back(r3);

    polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, curve, repo);
    }
    }*/

    //Spline Punkte aktualieren, bauen
    polynomialfit(LocalXForFirstCurve, LocalYForFirstCurve, degree, info, curve, repo);


return(curve);
}

COVERPLUGIN(NurbsSurface)
