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

        cout << "\nGoodbye particle " << i << endl;
    }
    particleList.clear();
    numParticles = 0;

    cout << "Goodbye BloodPlugin" << endl << endl;
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
    cover->setScale(1000);

    return true;
}

bool BloodPlugin::update() { 
    
    if(numParticles < 1) {
        return false;
    }

    osg::Matrix hand = cover -> getPointerMat();
    osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();

    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rotKnife;
    rotKnife.makeRotate(-osg::PI_2, osg::Vec3(0,0,1)); 

    //moving and drawing the knife
    knifeTransform -> setMatrix(rotKnife * handInObjectsRoot);

	//auto currentParticle = particleList.begin(); //iterator pointing to the first element of the particle list

    knife.currentPosition = knife.lengthInOR *  handInObjectsRoot;
    
    knife.shift = knife.currentPosition - knife.prevPosition;
    knife.prevPosition = knife.currentPosition;
	    
    static double prevKnifeSpeed;
    double knifeAccel;

    prevKnifeSpeed = 0;
    double knifeSpeed = knife.shift.length() / (cover -> frameDuration());
    knifeAccel = (knifeSpeed - prevKnifeSpeed) / cover -> frameDuration();
    prevKnifeSpeed = knifeSpeed;

    for(int currentParticleNumber = 0; currentParticleNumber < numParticles; currentParticleNumber++)
    {

	    if(knife.prevPosition.x() == 0) { //will be true on the first time update() is run
        	knife.prevPosition = knife.currentPosition;
        	particleList[currentParticleNumber] -> currentPosition = knife.currentPosition;
	    }

	    //**************************only animate the numParticle-1 element of the vector*******************************
	    if(particleList[currentParticleNumber] -> firstUpdate && particleList[currentParticleNumber] -> currentVelocity.length() != 0) {
        	particleList[currentParticleNumber] -> findReynoldsNum();
        	particleList[currentParticleNumber] -> findDragCoefficient();
        	particleList[currentParticleNumber] -> findTerminalVelocity();

        	if(numParticles == 2) cout << particleList[currentParticleNumber] -> terminalVelocity << endl;
        	particleList[currentParticleNumber] -> firstUpdate = false;
	    }


	    static double prevKnifeSpeed;
	    double knifeAccel;


	    //?: need a way to limit the position/velocity of the particle, can do terminal velocity/bounded z-position?
	    //*************************************************************************************
	    if(!particleList[currentParticleNumber] -> onKnife || 
	    (particleList[currentParticleNumber] -> currentVelocity.length() > 5/* 000 */ && 
	    particleList[currentParticleNumber] -> currentVelocity.length() < 50/* 000 */)) {
        	//particle leaves the knife with an initial velocity equal to the knife's velocity

        	cout << "particle " << numParticles << " has left the knife" << endl;
        	//cout << "numParticles is now: " << currentParticleNumber << endl;

        	//checking that velocity < terminal velocity
        	if(abs(particleList[currentParticleNumber] -> currentVelocity.x()) < particleList[currentParticleNumber] -> terminalVelocity && 
        	   abs(particleList[currentParticleNumber] -> currentVelocity.y()) < particleList[currentParticleNumber] -> terminalVelocity && 
        	   abs(particleList[currentParticleNumber] -> currentVelocity.z()) < particleList[currentParticleNumber] -> terminalVelocity) {

                	particleList[currentParticleNumber] -> currentVelocity += particleList[currentParticleNumber] -> gravity * 
                	double(cover -> frameDuration());

        	} else {
        	    cout << "v > term" << endl;

        	    if(numParticles == 2 && !isnan(particleList[currentParticleNumber] -> currentVelocity.x())) {
                	cout << "else\nvelocity: " << particleList[currentParticleNumber] -> currentVelocity << endl;
                	cout << "position: " << particleList[currentParticleNumber] -> currentPosition << endl;   
                	num++;

                	if(num == 2) {

                	    //while(true);
                	}

        	    } else if(isnan(particleList[currentParticleNumber] -> currentVelocity.x())) {
                	while(true);
        	    }

        	    particleList[currentParticleNumber] -> currentVelocity += 
        	    particleList[currentParticleNumber] -> gravity * double(cover -> frameDuration());

        	    if(abs(particleList[currentParticleNumber] -> currentVelocity.x()) >= 
        	    particleList[currentParticleNumber] -> terminalVelocity && 
        	    particleList[currentParticleNumber] -> currentVelocity.x() != 0) {

                	particleList[currentParticleNumber] -> currentVelocity.x() = 
                	(particleList[currentParticleNumber] -> currentVelocity.x() / 
                	abs(particleList[currentParticleNumber] -> currentVelocity.x())) * 
                	particleList[currentParticleNumber] -> terminalVelocity;
        	    }

        	    if(abs(particleList[currentParticleNumber] -> currentVelocity.y()) >= 
        	    particleList[currentParticleNumber] -> terminalVelocity && 
        	    particleList[currentParticleNumber] -> currentVelocity.y() != 0) {

                	particleList[currentParticleNumber] -> currentVelocity.y() = 
                	(particleList[currentParticleNumber] -> currentVelocity.y() / 
                	abs(particleList[currentParticleNumber] -> currentVelocity.y())) * 
                	particleList[currentParticleNumber] -> terminalVelocity;
        	    }

        	    if(abs(particleList[currentParticleNumber] -> currentVelocity.z()) >= 
        	    particleList[currentParticleNumber] -> terminalVelocity && 
        	    particleList[currentParticleNumber] -> currentVelocity.z() != 0) {

                	particleList[currentParticleNumber] -> currentVelocity.z() = 
                	(particleList[currentParticleNumber] -> currentVelocity.z() / 
                	abs(particleList[currentParticleNumber] -> currentVelocity.z())) * 
                	particleList[currentParticleNumber] -> terminalVelocity;
        	    }
        	}

		if(particleList[currentParticleNumber] -> currentPosition[2] < VRSceneGraph::instance()->floorHeight()/1000.0)
		{
		  // remote particle from active list
		  particleList[currentParticleNumber] -> currentVelocity = osg::Vec3(0,0,0);
		}
        	//no checks for terminal velocity
        	// particleList[currentParticleNumber] -> currentVelocity += particleList[currentParticleNumber] -> gravity * double(cover -> frameDuration());

        	particleList[currentParticleNumber] -> currentPosition += particleList[currentParticleNumber] -> currentVelocity * 
        	double(cover -> frameDuration());

        	particleList[currentParticleNumber] -> onKnife = false;

        	if(particleList[currentParticleNumber] -> currentVelocity.length() != 0) { //testing
        	    cout << "if-statement particle speed: " << particleList[currentParticleNumber] -> currentVelocity << endl; //testing
        	    cout << "if-statement particle position: " << particleList[currentParticleNumber] -> currentPosition << endl << endl;
        	}

	    } else {
        	//particle has the same velocity as the knife
        	particleList[currentParticleNumber] -> currentVelocity.set(knife.shift / double(cover -> frameDuration()));
        	particleList[currentParticleNumber] -> currentPosition.set(knife.currentPosition); 

        	if(particleList[currentParticleNumber] -> currentVelocity.length() != 0) {
        	    cout << "particle position: " << particleList[currentParticleNumber] -> currentPosition << endl;
        	    cout << "else-statement particle speed: " << particleList[currentParticleNumber] -> currentVelocity << endl << endl; //testing
        	}



	    }

		if(particleList[currentParticleNumber] -> currentPosition[2] < VRSceneGraph::instance()->floorHeight()/1000.0)
		{
		  // remote particle from active list
		}
	    //**************************************************************************************/
	    particleList[currentParticleNumber] -> matrix.makeTranslate(particleList[currentParticleNumber] -> currentPosition);
	    particleList[currentParticleNumber] -> bloodTransform -> setMatrix(particleList[currentParticleNumber] -> matrix);

	    //need a way to decrement numParticles once a particle goes out of bounds
	    // if(!particleList[currentParticleNumber] -> onKnife) {
	    //     numParticles--;
	    // }
    }
    
    return true;
}

void BloodPlugin::doAddBlood() { //this function causes seg faults
    //create a bunch of blood on the object
    //bloodJunks.push_back(new Blood()); //old code
    
    //numParticles++;
    particleList.push_back(new Droplet(osg::Vec4(0,0,1,1)));
    numParticles++;

    cover -> getObjectsRoot() -> addChild(particleList[numParticles - 1] -> bloodTransform);
    cout << "added new particle, number of existing particles is now: " << numParticles << endl;
    //??additional particles have nan as speed/position/cull matrix when animated?

    // cout << "still working on this" << endl << endl;
    // while(true);
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)
