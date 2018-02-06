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
 ** Author: F.Karle/ K.Ahmann	                                             **
 **                                                                          **
 ** History:  			  	                                                 **
 ** December 2017  v1		                                                 **
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

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>

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

static const int DEFAULT_REF = 30000000; 
static const int DEFAULT_MAX_REF = 100;

using namespace opencover;
using namespace std;
using namespace grmsg;

using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::coDirectory;


    void NurbsSurface::initUI()
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- NurbsSurface::initUI\n");

        SaveButton = new coButtonMenuItem("Save NurbsSurface");
        SaveButton-> setMenuListener(this);

        cover->getMenu()->add(SaveButton);

        tuiSaveTab = new coTUITab("Save NurbsSurface", coVRTui::instance()->mainFolder->getID());
        tuiSaveTab->setPos(0, 0);

        tuiFileNameLabel = new coTUILabel("Filename", tuiSaveTab->getID());
        tuiFileNameLabel->setPos(0, 2);

        tuiFileName = new coTUIEditField("Filename", tuiSaveTab->getID());
        tuiFileName->setEventListener(this);
        tuiFileName->setText("NurbsSurface.?");
        tuiFileName->setPos(1, 2);

        tuiSavedFileLabel = new coTUILabel("saved as:", tuiSaveTab->getID());
        tuiSavedFileLabel->setPos(2, 2);
    
        tuiSavedFile = new coTUILabel("", tuiSaveTab->getID());
        tuiSavedFile->setPos(3, 2);

        tuiSaveButton = new coTUIButton("Do it!", tuiSaveTab->getID());
        tuiSaveButton->setEventListener(this);
        tuiSaveButton->setPos(0, 4);  
    }

    bool NurbsSurface::init()
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- NurbsSurface::init\n");

        doInit = true;
        return true;
    }

    
    void NurbsSurface::tabletPressEvent(coTUIElement *tUIItem)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- NurbsSurface::tabletPressEvent\n");

        if (tUIItem == tuiSaveButton)
        {
            //preparesave
            doSave = true;
        }
    }

    
    void NurbsSurface::tabletReleaseEvent(coTUIElement *tUIItem)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- NurbsSurface::tabletReleaseEvent\n");
    }
    

/*    void NurbsSurface::menuEvent(coMenuItem *menuItem)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\n--- NurbsSurface::menuEvent\n");

        if (menuItem == SaveButton)
        {
            //prepareSave
            doSave = true;
        }
    }    
*/



NurbsSurface::NurbsSurface()
{

    const int num_points_u = 3; // number of points in the u parameter direction
    const int num_points_v = 3; // number of points in the v parameter direction

    double points[] = 
    {
        0, 0.005, 0,      0.03, 0.03, 0,      0.1, 0.05, 0,      
        -0.0025, 0, 0,      0.03, 0, 0.02,    0.1, 0, 0.05,      
        0, -0.005, 0,      0.03, -0.03, 0,     0.1, -0.05, 0,      
    };

    double u_par[] = {0, 1, 2}; // point parametrization in u-direction
    double v_par[] = {0, 1, 2}; // point parametrization in v-direction

    const int dim = 3; // dimension of the space we are working in

    const int num_surf = 1;

   
/*
    void addSliderButton(*Order_U, 2, 5, 5);
    void addSliderButton(*Order_V, 2, 5, 5);
*/
    const int order_u = 5;
    const int order_v = 5;

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
                order_u,   // the order of the generated surface in the 'u' parameter
                order_v,   // the order of the generated surface in the 'v' parameter
                1,            // open surface in the u direction
                1,            // open surface in the v direction 
                &result_surf, // the generated surface
                &jstat);      // status variable
        /*        if (jstat < 0) {
                  throw runtime_error("Error occured inside call to SISL routine s1537.");
                  } else if (jstat > 0) {
                  cerr << "WARNING: warning occured inside call to SISL routine s1537. \n"
                  << endl;
                  }*/
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

        //write to file
        osgDB::writeNodeFile(*geode, "test.obj");
        osgDB::writeNodeFile(*geode, "test.stl");

        freeSurf(result_surf);


        }
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

COVERPLUGIN(NurbsSurface)
