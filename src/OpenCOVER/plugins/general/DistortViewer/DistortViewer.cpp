/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

//OSG
#include <osg/Node>
#include <osg/Group>
#include <osg/Camera>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>

//Local
#include "DistortViewer.h"
#include "Settings.h"

//Covise
#include <cover/RenderObject.h>
#include <cover/coVRRenderer.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <config/CoviseConfig.h>

// Konstruktor
// wird aufgerufen wenn das Plugin gestartet wird
DistortViewer::DistortViewer()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "DistortViewer::DistortViewer\n");

    // Aktiviert in Opencover die Option, dass Scene in Textur gerendert wird
    // -> nur nötig, wenn DistortViewer aktiviert. Muss bei Start von OpenCover aktiviert sein
    VRViewer::instance()->setRenderToTexture(true);
}

// Destruktor
// wird aufgerufen wenn das Plugin zur Laufzeit beendet wird
DistortViewer::~DistortViewer()
{
    //Speicher freigeben
    fprintf(stderr, "DistortViewer::~DistortViewer\n");
}

// Initialisierung
// wird nach dem Konstruktor aufgerufen
bool DistortViewer::init()
{
    fprintf(stderr, "DistortViewer::init\n");

    Settings::getInstance()->loadFromXML();
    scene = new SceneVis();

    //Opencover neue, verzerrte Szene hinzufügen (einzigstes Child)
    cover->getScene()->addChild(scene->getSceneGroup(0)); //Hier Scene für Screen 0 darstellen

    //OpenGL errors debuggen
    if (coCoviseConfig::isOn("COVER.CheckForOpenGLErrors", false))
    {
        coVRConfig::instance()->channels[0].camera->getGraphicsContext()->getState()->setCheckForGLErrors(osg::State::ONCE_PER_ATTRIBUTE);
    }

    /*
	//--------------
	// Debug TEST
	//-------------
	// Variablen für Debugger
	float coverHeight = coVRConfig::instance()->screens[0].hsize;
	float coverWidth = coVRConfig::instance()->screens[0].vsize;
	osg::Vec3 coverPosXYZ = coVRConfig::instance()->screens[0].xyz;
	osg::Matrix coverProjMat = coVRConfig::instance()->screens[0].rightProj;
	osg::Matrix coverViewMat = coVRConfig::instance()->screens[0].rightView;

	osg::Vec3 eye, center, up;
	coverViewMat.getLookAt(eye, center, up);

	double c_left, c_right, c_bottom;
	double c_top, c_near, c_far;
	coverProjMat.getFrustum(c_left, c_right, c_bottom, c_top, c_near, c_far);
	//--------------------------*/

    return true;
}

// PreFrame function
// wird jedes mal aufgerufen bevor ein neuer Frame gerendert wird
void DistortViewer::preFrame()
{
    scene->updateScene(0);

    /*--------------
	// Debug TEST
	//-------------
	float coverHeight = coVRConfig::instance()->screens[0].hsize;
	float coverWidth = coVRConfig::instance()->screens[0].vsize;
	osg::Vec3 coverPosXYZ = coVRConfig::instance()->screens[0].xyz;
	osg::Matrix coverProjMat = coVRConfig::instance()->screens[0].rightProj;
	osg::Matrix coverViewMat = coVRConfig::instance()->screens[0].rightView;
	//----------------------------*/
}

bool DistortViewer::load()
{
    return true;
}

COVERPLUGIN(DistortViewer)
