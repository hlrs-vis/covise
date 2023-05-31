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
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/ui/Action.h>
#include <cover/ui/Menu.h>

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

BloodPlugin::BloodPlugin() 
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("Blood", cover->ui)
{
    fprintf(stderr, "BloodPlugin\n");
}

BloodPlugin *BloodPlugin::inst = nullptr;

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
    
    for(auto it = particlesOnGround.begin(); it != particlesOnGround.end(); it++) {
    	cover -> removeNode((*it) -> bloodTransform.get());
    	delete *it;
    }
  	particlesOnGround.clear();

    numParticles = 0;
}

bool BloodPlugin::init() {

    //*********************************************************************************************************************************INITIALIZING WEAPON CLASS
    //initialization
    knife.mass = 0.5;
    knife.timeElapsed = cover -> frameDuration();
    knife.prevPosition = osg::Vec3(0,0,0);
    knife.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).preMult(knife.hilt);
    knife.prevVelocity = osg::Vec3(0,0,0);
	knife.acceleration = osg::Vec3(0,0,0);
	
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
    hand = cover -> getPointerMat();
    handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();

    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rotKnife;
    rotKnife.makeRotate(-osg::PI_2, osg::Vec3(0,0,1));
    //moving and drawing the knife
    knifeTransform -> setMatrix(rotKnife * handInObjectsRoot);
	
	knife.start = knife.hilt * handInObjectsRoot;
    knife.end = knife.tip * handInObjectsRoot;
   
	auto thisParticle = particleList.begin(); 
	while(thisParticle != particleList.end()) {
	
		//finding the knife's velocity
		knife.currentPosition = knife.hilt * handInObjectsRoot;

		if(knife.prevPosition.length() == 0) { //will be true on the first time update() is run
		    knife.prevPosition = knife.currentPosition;
		}
		
		knife.shift = knife.currentPosition - knife.prevPosition;
		knife.prevPosition = knife.currentPosition;

		knife.currentVelocity = knife.shift / double(cover -> frameDuration());
		
		//finding the knife's acceleration
		if(knife.prevVelocity.length() == 0) {
			knife.prevVelocity = knife.currentVelocity;
		}
		
		knife.acceleration = (knife.currentVelocity - knife.prevVelocity) / double(cover -> frameDuration());
		knife.prevVelocity = knife.currentVelocity;
		
		if((*thisParticle) -> firstUpdate) {
			(*thisParticle) -> currentPosition = knife.currentPosition; //initialize the particle's position to be on the knife
			(*thisParticle) -> currentVelocity = knife.currentVelocity; //particle has same velocity as the knife
		
			if((*thisParticle) -> currentVelocity.length() != 0) {//do some mechanics calculations
				(*thisParticle) -> findReynoldsNum();
				(*thisParticle) -> findDragCoefficient();
				(*thisParticle) -> findTerminalVelocity();

				(*thisParticle) -> firstUpdate = false;
			}
		}

		if((!(*thisParticle) -> onKnife || ((*thisParticle) -> currentVelocity.length() > 5 && (*thisParticle) -> currentVelocity.length() < 50))) { //tests to see if the particle has left the knife
			
			//particle leaves the knife with an initial velocity equal to the knife's velocity
			//cout << "particle " << numParticles << " has left the knife" << endl;

			//checking that velocity < terminal velocity
			if(abs((*thisParticle) -> currentVelocity.x()) < (*thisParticle) -> terminalVelocity &&
			abs((*thisParticle) -> currentVelocity.y()) < (*thisParticle) -> terminalVelocity &&
			abs((*thisParticle) -> currentVelocity.z()) < (*thisParticle) -> terminalVelocity) {
				(*thisParticle) -> currentVelocity += (*thisParticle) -> gravity * double(cover -> frameDuration());

			} else { //if velocity > terminal velocity, set velocity of the component(s) to terminal velocity
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
				
				/*if((*thisParticle) -> currentVelocity.length() != 0) { //testing
					cout << "if-statement particle speed: " << (*thisParticle) -> currentVelocity << endl; //testing
					cout << "if-statement particle position: " << (*thisParticle) -> currentPosition << endl << endl;
				}*/
				
			} else {
			//particle has hit the floor, prevent it from moving
				(*thisParticle) -> currentVelocity = osg::Vec3(0,0,0);
				(*thisParticle) -> currentPosition.z() = VRSceneGraph::instance() -> floorHeight() / 1000.0;
				(*thisParticle) -> onFloor = true;
				(*thisParticle) -> onKnife = false;

				particlesOnGround.push_back(*thisParticle); //add this particle to the list of particles on the ground
				thisParticle = particleList.erase(thisParticle); //remove from the other list

				numParticles--;
				
				cout << "particle has hit the floor" << endl;
				continue; //skip past the animation stuff at the end of the while loop
			}

		} else { //particle is still on the knife
			(*thisParticle) -> acceleration = particleSlip(*thisParticle);
			
			cout << "a: " << (*thisParticle) -> acceleration << endl;
			cout << "a * time: " << (*thisParticle) -> acceleration * double(cover -> frameDuration()) << endl;
			
			(*thisParticle) -> prevPosition = (*thisParticle) -> currentPosition;
			
			(*thisParticle) -> currentVelocity = knife.currentVelocity + ((*thisParticle) -> acceleration) * double(cover -> frameDuration());

			(*thisParticle) -> currentPosition += (*thisParticle) -> currentVelocity * double(cover -> frameDuration());
			
			/*if((*thisParticle) -> prevPosition != (*thisParticle) -> currentPosition) { //testing
				cout << "else-statement particle speed: " << (*thisParticle) -> currentVelocity << endl;
				//cout << "else-statement particle position: " << (*thisParticle) -> currentPosition << endl << endl;
				
				cout << "prev pos: " << (*thisParticle) -> prevPosition << endl;
				cout << "current:  " << (*thisParticle) -> currentPosition << endl << endl;;
			} */
			
			if(knife.end < (*thisParticle) -> currentPosition) {
				(*thisParticle) -> onKnife = false;
			}
			
			/*if((*thisParticle) -> currentPosition.y() > knife.end.y() || (*thisParticle) -> currentPosition.z() < knife.end.z()) {
				(*thisParticle) -> onKnife = false;
			}*/
			
		}
		
		(*thisParticle) -> matrix.makeTranslate((*thisParticle) -> currentPosition);
		(*thisParticle) -> bloodTransform -> setMatrix((*thisParticle) -> matrix);
		thisParticle++;
	}

    return true;
}

void BloodPlugin::doAddBlood() {
    particleList.push_back(new Droplet(/*osg::Vec4(0,0,1,1)*/));
    numParticles++;

    cover -> getObjectsRoot() -> addChild(particleList.back() -> bloodTransform);
    cout << "added new particle, number of existing particles is now: " << numParticles << endl;
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

osg::Vec3 BloodPlugin::particleSlip(Droplet* p) {
	osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();
	osg::Matrix invHandInObjectsRoot = osg::Matrix::inverse(handInObjectsRoot);
	
	double staticFriction = GRAVITY * p -> mass * COEFF_STATIC_FRICTION;
	double kineticFriction = GRAVITY * p -> mass * COEFF_KINETIC_FRICTION;
	
	osg::Vec3 xAxis = osg::Vec3(handInObjectsRoot(0,0), handInObjectsRoot(0,1), handInObjectsRoot(0,2));
	osg::Vec3 yAxis = osg::Vec3(handInObjectsRoot(1,0), handInObjectsRoot(1,1), handInObjectsRoot(1,2));
	osg::Vec3 zAxis = osg::Vec3(handInObjectsRoot(2,0), handInObjectsRoot(2,1), handInObjectsRoot(2,2)); //3rd row of matrix is z-vector, use this as the normal for the subsequent calculations
	osg::Vec3 zAxisUnitVector = normalize(zAxis);
	
	osg::Vec3 normalVector = xAxis ^ yAxis;
	cout << "normal vector: " << normalVector << endl;
	osg::Plane knifePlane(normalVector, knife.currentPosition);
	double knifeParticleDistance = knifePlane.distance(p -> currentPosition);
	cout << "distance: " << knifeParticleDistance << endl;
	double restorativeFactor = 5.0;
	osg::Vec3 fRestorative = normalize(normalVector) * knifeParticleDistance * restorativeFactor;

	osg::Vec3 fApplied, fStaticFriction, fKineticFriction;
	osg::Vec3 fGravity = p -> gravity * p -> mass; //direction: -z
	//fNormal: projection of fGravity (multiply with z-axis unit vector) onto z-axis unit vector
	//osg::Vec3 fNormal = osg::Vec3(abs(fGravity.x()) * zAxisUnitVector.x(), abs(fGravity.y()) * zAxisUnitVector.y(), abs(fGravity.z()) * zAxisUnitVector.z()); //direction: perpendicular to knife's surface
	osg::Vec3 fNet;
	
	osg::Vec3 veloInPlane = osg::Matrix::transform3x3(knife.currentVelocity,invHandInObjectsRoot);
	veloInPlane.z() = 0;
	veloInPlane = osg::Matrix::transform3x3(veloInPlane,handInObjectsRoot);
	
	fApplied = (veloInPlane * p -> mass * COEFF_STATIC_FRICTION)/cover -> frameDuration();
	
	fNet = /*fGravity + */fRestorative/* + fApplied*/;
	return fNet / p -> mass;
	
	/*cout << "gravity: " << fGravity << endl;
	cout << "applied: " << fApplied << endl;
	cout << "normal:  " << fNormal << endl;
	cout << "static:  " << fStaticFriction << endl;
	cout << "kinetic: " << fKineticFriction << endl;
	cout << "net:     " << fNet << endl << endl;*/
	
	return fNet / p -> mass;
}

/*if(fApplied.length() != 0) {
		fStaticFriction = -normalize(veloInPlane) * staticFriction; //direction: opposes motion
		fKineticFriction = -normalize(veloInPlane) * kineticFriction; //direction: opposes motion
	} else {
		fStaticFriction = osg::Vec3(0,0,0); //direction: opposes motion
		fKineticFriction = osg::Vec3(0,0,0); //direction: opposes motion;
	}
	
	fNet = fGravity + fNormal;
	
	if(fStaticFriction < fApplied) { //applied force > static friction
		fNet += fApplied + fKineticFriction;
	} else {
		fNet += fStaticFriction;
	}*/
	
COVERPLUGIN(BloodPlugin)
