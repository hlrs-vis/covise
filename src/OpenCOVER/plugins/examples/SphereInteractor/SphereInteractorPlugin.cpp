/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2010 HLRS  **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: A.Gottlieb		                                             **
 **                                                                          **
 ** History:  								     **
 ** Nov-10  v1	    				       		             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "SphereInteractorPlugin.h"
#include <cover/RenderObject.h>

SphereInteractorPlugin::SphereInteractorPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "SphereInteractorPlugin::SphereInteractorPlugin\n");
    //Create new coTrackerButtonInteraction Object
    myinteraction = new coTrackerButtonInteraction(coInteraction::ButtonA, "MoveMode", coInteraction::Medium);
    interActing = false;

    //Creating an osg::Sphere and colorize it red
    //at center Radius 1
    mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
    //red color
    mySphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
    osg::Geode *myGeode = new osg::Geode();
    osg::StateSet *mystateSet = myGeode->getOrCreateStateSet();
    setStateSet(mystateSet);

    //Creating Sensor for sphere, so that interactor and geometry are attached
    aSensor = new mySensor(myGeode, "mySensor", myinteraction, mySphereDrawable);
    sensorList.append(aSensor);
    myGeode->setName("mySphere");
    myGeode->addDrawable(mySphereDrawable);
    mymtf = new osg::MatrixTransform();
    //init position
    mymtf->setMatrix(mymtf->getMatrix().translate(osg::Vec3(0, 0, 0)));

    mymtf->addChild(myGeode);

    cover->getObjectsRoot()->addChild(mymtf);
}

//-----------------------------------------------------------
// this is called if the plugin is removed at runtime
SphereInteractorPlugin::~SphereInteractorPlugin()
{
    fprintf(stderr, "SphereInteractorPlugin::~SphereInteractorPlugin\n");
    if (sensorList.find(aSensor))
        sensorList.remove();
    delete aSensor;
}

//-----------------------------------------------------------
void
SphereInteractorPlugin::preFrame()
{
    sensorList.update();
    //Test if button is pressed
    int state = cover->getPointerButton()->getState();
    if (myinteraction->isRunning()) //when interacting the Sphere will be moved
    {
        static osg::Matrix invStartHand;
        static osg::Matrix startPos;
        if (!interActing)
        {
            //remember invStartHand-Matrix, when interaction started and mouse button was pressed
            invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat());
            startPos = mymtf->getMatrix(); //remember position of sphere, when interaction started
            interActing = true; //register interaction
        }
        else
        {
            //calc the tranformation matrix when interacting is running and mouse button was pressed
            osg::Matrix transMat = startPos * invStartHand * (cover->getPointerMat() * cover->getInvBaseMat());
            mymtf->setMatrix(transMat);
        }
    }
    if (myinteraction->wasStopped() && state == false)
    {
        interActing = false; //unregister interaction
    }
}

//-----------------------------------------------------------
void SphereInteractorPlugin::setStateSet(osg::StateSet *stateSet)
{
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::DIFFUSE);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    osg::LightModel *defaultLm;
    defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
mySensor::mySensor(osg::Node *node, std::string name, coTrackerButtonInteraction *_interactionA, osg::ShapeDrawable *cSphDr)
    : coPickSensor(node)
{
    sensorName = name;
    isActive = false;
    _interA = _interactionA;
    shapDr = cSphDr;
}

mySensor::~mySensor()
{
}

//-----------------------------------------------------------
void mySensor::activate()
{
    isActive = true;
    cout << "---Activate--" << sensorName.c_str() << endl;
    coInteractionManager::the()->registerInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 1., 0., 1.0f));
}

//-----------------------------------------------------------
void mySensor::disactivate()
{
    cout << "---Disactivate--" << sensorName.c_str() << endl;
    isActive = false;
    coInteractionManager::the()->unregisterInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 0., 0., 1.0f));
}

//-----------------------------------------------------------

std::string mySensor::getSensorName()
{
    return sensorName;
}

//-----------------------------------------------------------
bool mySensor::isSensorActive()
{
    if (isActive)
        return true;
    else
        return false;
}

//-----------------------------------------------------------

COVERPLUGIN(SphereInteractorPlugin)
