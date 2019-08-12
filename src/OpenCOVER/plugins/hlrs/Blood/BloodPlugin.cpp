/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: BloodPlugin OpenCOVER Plugin (is polite)                    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** June 2008  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "BloodPlugin.h"
#include "globals.h"
#include <cover/VRSceneGraph.h>

using namespace opencover;

/******************************************CARTESIAN COORDINATE SYSTEM FOR OPENCOVER************************************
					z
					z
					z
					z
					z
					z
					z
					xxxxxxxxxxxxxxxxxxxxxxx
- positive y goes into the screen
************************************************************************************************************************/

bool once = true; //testing
int num = 0; //testing

//bool firstUpdate = true;

//defines the transformation matrix in x,y,z world coordinates and sets artificial 4th coordinate to 1
double t[] = {1.,0.,0.,0., //x scaling
              0.,1.,0.,0., //y scaling
              0.,0.,1.,0., //z scaling
              1.,1.,1.,1.}; //translation operation in x/y/z direction

BloodPlugin::BloodPlugin() : ui::Owner("Blood", cover->ui)
{
    fprintf(stderr, "BloodPlugin\n");
}

BloodPlugin *BloodPlugin::inst = nullptr; // never reassigned to a different value(???)

// this is called if the plugin is removed at runtime
BloodPlugin::~BloodPlugin()
{
    for (auto junk = bloodJunks.begin(); junk != bloodJunks.end(); junk++)
    {
        delete *junk;
    }
    bloodJunks.clear();

    for(auto it = particleList.begin(); it != particleList.end(); it++) {
    	cover -> removeNode((*it) -> bloodTransform.get());
    	delete *it;
    }
    particleList.clear();

    numParticles = 0;
}

bool BloodPlugin::init() {

    //*********************************************************************************************************************************INITIALIZING WEAPON CLASS
    //initialization
    knife.mass = 0.5;
    knife.timeElapsed = cover -> frameDuration();
    knife.prevPosition = osg::Vec3(0,0,0);
    knife.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).preMult(knife.hilt/*knife.tip*/);

    //****************************************************************************************************************************DRAWING THE KNIFE
    //loading an external 3D knife model
    knife.knifePtr = coVRFileManager::instance()->loadIcon("knife"); //loads 3d model of a knife into OpenCOVER
    knife.knifePtr -> setName("KnifePtr");

    //matrix controlling the movement of the knife, eventually should be the same as pointer matrix/hand matrix
    knifeTransform = new osg::MatrixTransform;

    knifeBaseTransform.set(t);
    knifeTransform -> setMatrix(knifeBaseTransform);
    knifeTransform -> addChild(knife.knifePtr);
    knifeTransform -> setName("knifeTransform");

    //********************************RENDERING THE KNIFE IN OPENCOVER************************************
    if(knife.knifePtr) { //check if the knife image loaded properly, if so, add it to the graph
        cover -> getObjectsRoot() -> addChild(knifeTransform);
   		cout << "Hello Knife" << endl;
   		cout << "Hello Knife Animation" << endl;
    }

    //*********************************************************************************************************************************INITIALIZING DROPLET CLASS
    //initializing things for the particle
    numParticles = 1;
    
    particleList.push_back(new Droplet());
    cover -> getObjectsRoot() -> addChild(particleList.front() -> bloodTransform);

	cout << "Hello Blood" << endl;
    //thisParticle = particleList.begin();

    //*****************************************************************************************************************CREATING THE BLOOD PLUGIN UI MENU
    bloodMenu = new ui::Menu("Blood", this);
    bloodMenu->setText("Blood");

    addBlood = new ui::Action(bloodMenu, "addBlood");
    addBlood->setText("Add Blood");
    addBlood->setCallback([this]() {doAddBlood(); });
	
	cover -> setScale(1000);
    return true;
}

bool BloodPlugin::update() {
    osg::Matrix hand = cover -> getPointerMat();
    osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();

    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rotKnife;
    rotKnife.makeRotate(-osg::PI_2, osg::Vec3(0,0,1));

    //moving and drawing the knife
    knifeTransform -> setMatrix(rotKnife * handInObjectsRoot);

    //knife.currentPosition = knife.tip * handInObjectsRoot;
    knife.currentPosition = knife.hilt * handInObjectsRoot;

    if(knife.prevPosition.length() == 0) { //will be true on the first time update() is run
        knife.prevPosition = knife.currentPosition;
    }

	knife.shift = knife.currentPosition - knife.prevPosition;
    knife.prevPosition = knife.currentPosition;

	knife.currentVelocity = knife.shift / double(cover -> frameDuration());

	auto thisParticle = particleList.begin(); 
	while(thisParticle != particleList.end()) {

		if((*thisParticle) -> firstUpdate) { //if update() is entered for the first time, calculate the necessary mechanics numbers
			(*thisParticle) -> currentPosition = knife.currentPosition;

			if((*thisParticle) -> currentVelocity.length() != 0) {
				(*thisParticle) -> findReynoldsNum();
				(*thisParticle) -> findDragCoefficient();
				(*thisParticle) -> findTerminalVelocity();

				(*thisParticle) -> firstUpdate = false;
			}
		}

		//for finding knife acceleration, can be used for friction and other forces
		if((*thisParticle) -> onKnife) {
			if(knife.prevVelocity.length() == 0) {
				knife.prevVelocity = knife.currentVelocity;
			}
			
			knife.acceleration = (knife.currentVelocity - knife.prevVelocity) / double(cover -> frameDuration());
			knife.prevVelocity = knife.currentVelocity;
		}

		if((!(*thisParticle) -> onKnife || ((*thisParticle) -> currentVelocity.length() > 5 && (*thisParticle) -> currentVelocity.length() < 50))) { //tests to see if the particle has left the knife
			//particle leaves the knife with an initial velocity equal to the knife's velocity
			cout << "particle " << numParticles << " has left the knife" << endl;

			//checking that velocity < terminal velocity
			if(abs((*thisParticle) -> currentVelocity.x()) < (*thisParticle) -> terminalVelocity &&
			abs((*thisParticle) -> currentVelocity.y()) < (*thisParticle) -> terminalVelocity &&
			abs((*thisParticle) -> currentVelocity.z()) < (*thisParticle) -> terminalVelocity) {
				(*thisParticle) -> currentVelocity += (*thisParticle) -> gravity * double(cover -> frameDuration());

			} else { //velocity > terminal velocity
				cout << "v > term" << endl;

				(*thisParticle) -> currentVelocity += (*thisParticle) -> gravity * double(cover -> frameDuration());

				if(abs((*thisParticle) -> currentVelocity.x()) >= (*thisParticle) -> terminalVelocity) {
					(*thisParticle) -> currentVelocity.x() = signOf((*thisParticle) -> currentVelocity.x()) * (*thisParticle) -> terminalVelocity;
				}

				if(abs((*thisParticle) -> currentVelocity.y()) >= (*thisParticle) -> terminalVelocity) {
					(*thisParticle) -> currentVelocity.y() = signOf((*thisParticle) -> currentVelocity.y()) * (*thisParticle) -> terminalVelocity;
				}

				if(abs((*thisParticle) -> currentVelocity.z()) >= (*thisParticle) -> terminalVelocity) {
					(*thisParticle) -> currentVelocity.z() = signOf((*thisParticle) -> currentVelocity.z()) * (*thisParticle) -> terminalVelocity;
				}
			}

			//no checks for terminal velocity
			// (*thisParticle) -> currentVelocity += (*thisParticle) -> gravity * double(cover -> frameDuration());

			// check if particle is below the floorHeight in the CAVE
			if((*thisParticle) -> currentPosition.z() > VRSceneGraph::instance() -> floorHeight() / 1000.0) { //divide by 1000 since floorHeight is in mm
				(*thisParticle) -> currentPosition += (*thisParticle) -> currentVelocity * double(cover -> frameDuration());
				(*thisParticle) -> onKnife = false;
				
				if((*thisParticle) -> currentVelocity.length() != 0) { //testing
					cout << "if-statement particle speed: " << (*thisParticle) -> currentVelocity << endl; //testing
					cout << "if-statement particle position: " << (*thisParticle) -> currentPosition << endl << endl;
				}
				
			} else {
				(*thisParticle) -> currentVelocity = osg::Vec3(0,0,0);
				(*thisParticle) -> currentPosition.z() = VRSceneGraph::instance() -> floorHeight() / 1000.0;
				(*thisParticle) -> onFloor = true;
				(*thisParticle) -> onKnife = false;

				particlesOnGround.push_front(*thisParticle);
				thisParticle = particleList.erase(thisParticle);

				numParticles--;
				
				cout << "particle has hit the floor" << endl;
				continue;
			}

		} else { //particle is still on the knife
			(*thisParticle) -> acceleration = particleSlip(*thisParticle);

			(*thisParticle) -> currentVelocity = knife.currentVelocity + (*thisParticle) -> acceleration * double(cover -> frameDuration());
			(*thisParticle) -> currentPosition = knife.currentPosition + (*thisParticle) -> currentVelocity * double(cover -> frameDuration());

			if((*thisParticle) -> currentVelocity.length() != 0) { //testing
				cout << "else-statement particle speed: " << (*thisParticle) -> currentVelocity << endl; //testing
				cout << "else-statement particle position: " << (*thisParticle) -> currentPosition << endl << endl;
			}
		}
		
		(*thisParticle) -> matrix.makeTranslate((*thisParticle) -> currentPosition);
		(*thisParticle) -> bloodTransform -> setMatrix((*thisParticle) -> matrix);
		thisParticle++;
	}

    return true;
}

void BloodPlugin::doAddBlood() {
    particleList.push_back(new Droplet(osg::Vec4(0,0,1,1)));
    numParticles++;

    cover -> getObjectsRoot() -> addChild(particleList.back() -> bloodTransform);
    cout << "added new particle, number of existing particles is now: " << numParticles << endl;
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

osg::Vec3 BloodPlugin::particleSlip(Droplet* p) {
	double sinAngleOfInclination = knife.currentPosition.z()/*hypot(knife.currentPosition.x(), knife.currentPosition.y())*/ / knife.currentPosition.length();
	
	//frictional forces are a scalar with unit vector in the opposite direction as movement
	double sF = -GRAVITY * sinAngleOfInclination * p -> mass * COEFF_STATIC_FRICTION;
	double kF = -GRAVITY * sinAngleOfInclination * p -> mass * COEFF_KINETIC_FRICTION;
	
	osg::Vec3 grav = p -> gravity * p -> mass;
	osg::Vec3 forceApplied = knife.acceleration * knife.mass;
	cout << "force applied: " << forceApplied << endl;

	osg::Vec3 posForce = forceApplied + grav;
	cout << "positive forces: " << posForce << endl;
	osg::Vec3 posForceUnitVector = posForce / posForce.length();
	
	osg::Vec3 staticFriction = -posForceUnitVector * sF;
	cout << "static friction: " << staticFriction << endl;
	osg::Vec3 kineticFriction = -posForceUnitVector * kF;
	cout << "kinetic friction: " << kineticFriction << endl;
 	
	osg::Vec3 netForce;
	osg::Vec3 a;

	if(posForce < staticFriction) {
		netForce = posForce + staticFriction;
	} else if (posForce == staticFriction) {
		netForce = osg::Vec3(0,0,0);
	} else {
		netForce = posForce + kineticFriction;
	}
	cout << "net force: " << netForce << endl;
	
	a = netForce / p -> mass;
	cout << "a: " << a << endl << endl;
	return a;
	
}

COVERPLUGIN(BloodPlugin)
