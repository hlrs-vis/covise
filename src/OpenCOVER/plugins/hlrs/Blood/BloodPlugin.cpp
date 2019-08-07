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

    for(int i = 0; i < numParticles; i++) {
        //remove node to delete from the scene graph - not sure which one to use?
        cover -> removeNode(particleList[i] -> bloodTransform.get());
        //cover -> getObjectsRoot() -> removeChild(particleList[i] -> bloodTransform.get());

        delete particleList[i];
        particleList[i] = NULL;
    }
    particleList.clear();

    numParticles = 0;
}

bool BloodPlugin::init() {

    //*********************************************************************************************************************************INITIALIZING WEAPON CLASS
    //initialization
    knife.timeElapsed = cover -> frameDuration();
    knife.prevPosition = osg::Vec3(0,0,0);
    knife.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).preMult(knife.lengthInOR);

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
    numParticles++;
    particleList.resize(numParticles);

    for(int i = 0; i < numParticles; i++) {
        particleList[i] = new Droplet();
        cover -> getObjectsRoot() -> addChild(particleList[i] -> bloodTransform);

    	cout << "Hello Blood" << endl;
    }

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

    knife.currentPosition = knife.lengthInOR *  handInObjectsRoot;

    if(knife.prevPosition.length() == 0) { //will be true on the first time update() is run
        knife.prevPosition = knife.currentPosition;
    }

	knife.shift = knife.currentPosition - knife.prevPosition;
    knife.prevPosition = knife.currentPosition;

	knife.velocity = knife.shift / double(cover -> frameDuration());

	for(int thisParticle = 0; thisParticle < numParticles; thisParticle++) { //loop through all particles in the vector

		if(particleList[thisParticle] -> firstUpdate) {
			particleList[thisParticle] -> currentPosition = knife.currentPosition;

			if(particleList[thisParticle] -> currentVelocity.length() != 0) {
				particleList[thisParticle] -> findReynoldsNum();
				particleList[thisParticle] -> findDragCoefficient();
				particleList[thisParticle] -> findTerminalVelocity();

				particleList[thisParticle] -> firstUpdate = false;
			}
		}

		//for finding knife acceleration, can be used for friction and other forces
		if(particleList[thisParticle] -> onKnife) {
			if(knife.prevSpeed = 0) {
				knife.prevSpeed = knife.currentSpeed;
			}

			knife.currentSpeed = knife.velocity.length();
			knife.accel = (knife.currentSpeed - knife.prevSpeed) / double(cover -> frameDuration());
			knife.prevSpeed = knife.currentSpeed;
		}

		if(/*!particleList[thisParticle] -> onFloor && */(!particleList[thisParticle] -> onKnife || 
		(particleList[thisParticle] -> currentVelocity.length() > 5 && particleList[thisParticle] -> currentVelocity.length() < 50))) {
			//particle leaves the knife with an initial velocity equal to the knife's velocity
			cout << "particle " << numParticles << " has left the knife" << endl;

			//checking that velocity < terminal velocity
			if(abs(particleList[thisParticle] -> currentVelocity.x()) < particleList[thisParticle] -> terminalVelocity &&
			abs(particleList[thisParticle] -> currentVelocity.y()) < particleList[thisParticle] -> terminalVelocity &&
			abs(particleList[thisParticle] -> currentVelocity.z()) < particleList[thisParticle] -> terminalVelocity) {
				particleList[thisParticle] -> currentVelocity += particleList[thisParticle] -> gravity * double(cover -> frameDuration());

			} else { //velocity > terminal velocity
				cout << "v > term" << endl;

				particleList[thisParticle] -> currentVelocity += particleList[thisParticle] -> gravity * double(cover -> frameDuration());

				if(abs(particleList[thisParticle] -> currentVelocity.x()) >= particleList[thisParticle] -> terminalVelocity) {
					particleList[thisParticle] -> currentVelocity.x() = signOf(particleList[thisParticle] -> currentVelocity.x()) * particleList[thisParticle] -> terminalVelocity;
				}

				if(abs(particleList[thisParticle] -> currentVelocity.y()) >= particleList[thisParticle] -> terminalVelocity) {
					particleList[thisParticle] -> currentVelocity.y() = signOf(particleList[thisParticle] -> currentVelocity.y()) * particleList[thisParticle] -> terminalVelocity;
				}

				if(abs(particleList[thisParticle] -> currentVelocity.z()) >= particleList[thisParticle] -> terminalVelocity) {
					particleList[thisParticle] -> currentVelocity.z() = signOf(particleList[thisParticle] -> currentVelocity.z()) * particleList[thisParticle] -> terminalVelocity;
				}
			}

			//no checks for terminal velocity
			// particleList[thisParticle] -> currentVelocity += particleList[thisParticle] -> gravity * double(cover -> frameDuration());

			// check if particle is below the floorHeight in the CAVE, if it is, don't change the position anymore
			// ******************can't tell if this check messes up a bunch of features or not bc the particle drops??? but the velocity is messed up maybe???
			if(particleList[thisParticle] -> currentPosition.z() > VRSceneGraph::instance() -> floorHeight() / 1000.0) {
				particleList[thisParticle] -> currentPosition += particleList[thisParticle] -> currentVelocity * double(cover -> frameDuration());

			} else {
				particleList[thisParticle] -> currentVelocity = osg::Vec3(0,0,0);
				particleList[thisParticle] -> currentPosition.z() = VRSceneGraph::instance() -> floorHeight() / 1000.0;
				particleList[thisParticle] -> onFloor = true;
				cout << "particle has hit the floor" << endl;
			}

			particleList[thisParticle] -> onKnife = false;

			if(particleList[thisParticle] -> currentVelocity.length() != 0) { //testing
				cout << "if-statement particle speed: " << particleList[thisParticle] -> currentVelocity << endl; //testing
				cout << "if-statement particle position: " << particleList[thisParticle] -> currentPosition << endl << endl;
			}

		} else { //particle is still on the knife
			//particle has the same velocity as the knife
			particleList[thisParticle] -> currentVelocity.set(knife.velocity);
			particleList[thisParticle] -> currentPosition.set(knife.currentPosition);

			if(particleList[thisParticle] -> currentVelocity.length() != 0) {
				cout << "else-statement particle speed: " << particleList[thisParticle] -> currentVelocity << endl << endl; //testing
				cout << "particle position: " << particleList[thisParticle] -> currentPosition << endl;
			}

		}

		//matrix transformations to draw the new position
		particleList[thisParticle] -> matrix.makeTranslate(particleList[thisParticle] -> currentPosition);
		particleList[thisParticle] -> bloodTransform -> setMatrix(particleList[thisParticle] -> matrix);

	//for loop should end here
	}

    return true;
}

void BloodPlugin::doAddBlood() {
    particleList.push_back(new Droplet(osg::Vec4(0,0,1,1)));
    numParticles++;

    cover -> getObjectsRoot() -> addChild(particleList[numParticles - 1] -> bloodTransform); //keep this as numParticles-1
    cout << "added new particle, number of existing particles is now: " << numParticles << endl;
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)
