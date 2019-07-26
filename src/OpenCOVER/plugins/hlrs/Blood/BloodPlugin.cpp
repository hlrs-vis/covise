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

    knife.currentPosition = knife.lengthInOR *  handInObjectsRoot;

    if(knife.prevPosition.x() == 0) { //will be true on the first time update() is run
        knife.prevPosition = knife.currentPosition;
        particleList[numParticles - 1] -> currentPosition = knife.currentPosition;
    }
    
    //**************************only animate the numParticle-1 element of the vector*******************************
    if(particleList[numParticles - 1] -> firstUpdate && particleList[numParticles - 1] -> currentVelocity.length() != 0) {
        particleList[numParticles - 1] -> findReynoldsNum();
        particleList[numParticles - 1] -> findDragCoefficient();
        particleList[numParticles - 1] -> findTerminalVelocity();

        if(numParticles == 2) cout << particleList[numParticles - 1] -> terminalVelocity << endl;
        particleList[numParticles - 1] -> firstUpdate = false;
    }
    
    knife.shift = knife.currentPosition - knife.prevPosition;
    knife.prevPosition = knife.currentPosition;

    static double prevKnifeSpeed;
    double knifeAccel;

    if(particleList[numParticles - 1] -> onKnife) {
        prevKnifeSpeed = 0;
        double knifeSpeed = knife.shift.length() / (cover -> frameDuration());
        knifeAccel = (knifeSpeed - prevKnifeSpeed) / cover -> frameDuration();
        prevKnifeSpeed = knifeSpeed;
    }
    
    //?: need a way to limit the position/velocity of the particle, can do terminal velocity/bounded z-position?
    //*************************************************************************************
    if(!particleList[numParticles - 1] -> onKnife || 
    (particleList[numParticles - 1] -> currentVelocity.length() > 5/* 000 */ && 
    particleList[numParticles - 1] -> currentVelocity.length() < 50/* 000 */)) {
        //particle leaves the knife with an initial velocity equal to the knife's velocity
        
        cout << "particle " << numParticles << " has left the knife" << endl;
        //cout << "numParticles is now: " << numParticles - 1 << endl;
        
       /*  if(numParticles == 2 && !isnan(particleList[numParticles - 1] -> currentVelocity.x() )) {
            cout << "velocity: " << particleList[numParticles - 1] -> currentVelocity << endl;
            cout << "position: " << particleList[numParticles - 1] -> currentPosition << endl;   
            
        } else  */if(isnan(particleList[numParticles - 1] -> currentVelocity.x())) {
            while(true);
        }

        //checking that velocity < terminal velocity
        if(abs(particleList[numParticles - 1] -> currentVelocity.x()) < particleList[numParticles - 1] -> terminalVelocity && 
           abs(particleList[numParticles - 1] -> currentVelocity.y()) < particleList[numParticles - 1] -> terminalVelocity && 
           abs(particleList[numParticles - 1] -> currentVelocity.z()) < particleList[numParticles - 1] -> terminalVelocity) {

                particleList[numParticles - 1] -> currentVelocity += particleList[numParticles - 1] -> gravity * 
                double(cover -> frameDuration());

                if(numParticles == 2 && !isnan(particleList[numParticles - 1] -> currentVelocity.x() )) {
                    cout << "if\nvelocity: " << particleList[numParticles - 1] -> currentVelocity << endl;
                    cout << "position: " << particleList[numParticles - 1] -> currentPosition << endl; 
                    //while(true);  
                }
                
        } else {
            cout << "v > term" << endl;
    
            if(numParticles == 2 && !isnan(particleList[numParticles - 1] -> currentVelocity.x())) {
                cout << "else\nvelocity: " << particleList[numParticles - 1] -> currentVelocity << endl;
                cout << "position: " << particleList[numParticles - 1] -> currentPosition << endl;   
                num++;

                if(num == 2) {
                    
                    //while(true);
                }
                
            } else if(isnan(particleList[numParticles - 1] -> currentVelocity.x())) {
                while(true);
            }

            //? when new particle is added via doAddBlood, the velocity of the particle at this point is always 0?

            particleList[numParticles - 1] -> currentVelocity += 
            particleList[numParticles - 1] -> gravity * double(cover -> frameDuration());

            if(abs(particleList[numParticles - 1] -> currentVelocity.x()) >= 
            particleList[numParticles - 1] -> terminalVelocity && 
            particleList[numParticles - 1] -> currentVelocity.x() != 0) {

                particleList[numParticles - 1] -> currentVelocity.x() = 
                (particleList[numParticles - 1] -> currentVelocity.x() / 
                abs(particleList[numParticles - 1] -> currentVelocity.x())) * 
                particleList[numParticles - 1] -> terminalVelocity;
            }

            if(abs(particleList[numParticles - 1] -> currentVelocity.y()) >= 
            particleList[numParticles - 1] -> terminalVelocity && 
            particleList[numParticles - 1] -> currentVelocity.y() != 0) {

                particleList[numParticles - 1] -> currentVelocity.y() = 
                (particleList[numParticles - 1] -> currentVelocity.y() / 
                abs(particleList[numParticles - 1] -> currentVelocity.y())) * 
                particleList[numParticles - 1] -> terminalVelocity;
            }

            if(abs(particleList[numParticles - 1] -> currentVelocity.z()) >= 
            particleList[numParticles - 1] -> terminalVelocity && 
            particleList[numParticles - 1] -> currentVelocity.z() != 0) {

                particleList[numParticles - 1] -> currentVelocity.z() = 
                (particleList[numParticles - 1] -> currentVelocity.z() / 
                abs(particleList[numParticles - 1] -> currentVelocity.z())) * 
                particleList[numParticles - 1] -> terminalVelocity;
            }
        }
        
        //no checks for terminal velocity
        // particleList[numParticles - 1] -> currentVelocity += particleList[numParticles - 1] -> gravity * double(cover -> frameDuration());
        
        particleList[numParticles - 1] -> currentPosition += particleList[numParticles - 1] -> currentVelocity * 
        double(cover -> frameDuration());

        particleList[numParticles - 1] -> onKnife = false;
            
        if(particleList[numParticles - 1] -> currentVelocity.length() != 0) { //testing
            cout << "if-statement particle speed: " << particleList[numParticles - 1] -> currentVelocity << endl; //testing
            cout << "if-statement particle position: " << particleList[numParticles - 1] -> currentPosition << endl << endl;
        }

    } else {
        //particle has the same velocity as the knife
        particleList[numParticles - 1] -> currentVelocity.set(knife.shift / double(100*cover -> frameDuration()));
        particleList[numParticles - 1] -> currentPosition.set(knife.currentPosition); 
        
        if(particleList[numParticles - 1] -> currentVelocity.length() != 0) {
            cout << "particle position: " << particleList[numParticles - 1] -> currentPosition << endl;
            cout << "else-statement particle speed: " << particleList[numParticles - 1] -> currentVelocity << endl << endl; //testing
        }

    }
    //**************************************************************************************/
    particleList[numParticles - 1] -> matrix.makeTranslate(particleList[numParticles - 1] -> currentPosition);
    particleList[numParticles - 1] -> bloodTransform -> setMatrix(particleList[numParticles - 1] -> matrix);

    //need a way to decrement numParticles once a particle goes out of bounds
    // if(!particleList[numParticles - 1] -> onKnife) {
    //     numParticles--;
    // }

    return true;
}

void BloodPlugin::doAddBlood() { //this function causes seg faults
    //create a bunch of blood on the object
    //bloodJunks.push_back(new Blood()); //old code
    
    //numParticles++;
    particleList.push_back(new Droplet(numParticles));
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