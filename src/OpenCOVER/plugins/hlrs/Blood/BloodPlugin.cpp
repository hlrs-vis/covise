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

bool firstUpdate = true;

BloodPlugin::BloodPlugin() : ui::Owner("Blood", cover->ui)
{
    fprintf(stderr, "BloodPlugin\n");
}

BloodPlugin *BloodPlugin::inst=nullptr;

// this is called if the plugin is removed at runtime
BloodPlugin::~BloodPlugin()
{
    for (auto junk = bloodJunks.begin(); junk != bloodJunks.end(); junk++)
    {
        delete *junk;
    }
    bloodJunks.clear();
}

//defines the transformation matrix in x,y,z world coordinates and sets artificial 4th coordinate to 1
double t[] = {1.,0.,0.,0., //x scaling
              0.,1.,0.,0., //y scaling
              0.,0.,1.,0., //z scaling
              1.,1.,1.,1.}; //translation operation in x/y/z direction

bool BloodPlugin::init() {
    // particlePosition.set(0,0,0);
    // velocity.set(10,0,100);
    // acc.set(0,0,-10);
    
    //*********************************************************************************************************************************INITIALIZING WEAPON CLASS
    //initialization
    knife.timeElapsed = cover -> frameDuration();
    knife.prevPosition = osg::Vec3(0,0,0);
    knife.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).preMult(knife.lengthInOR);
    //osg::Matrix hand = cover -> getPointerMat(); //convert pointerMat to ObjectsRoot coordinates
    //knife.currentPosition = hand.postMult(knife.length); //4x4 hand matrix * 3x1 length vector = 3x1 vector of the x/y/z coordinate of the tip of the knife
    
    //****************************************************************************************************************************DRAWING THE KNIFE	  
    //loading an external 3D knife model
    knife.knifePtr = coVRFileManager::instance()->loadIcon("knife"); //loads 3d model of a knife into OpenCOVER
    knife.knifePtr -> setName("KnifePtr");
    
    //matrix controlling the movement of the knife, eventually should be the same as pointer matrix/hand matrix
    knifeTransform = new osg::MatrixTransform;
   
   //???knife is currently positioned in -x direcion, multiply it to transform to y-direction
   //try makeRotate and rotate it 90 degrees clockwise (90 degrees about z-axis)
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
    particle.radius = 0.001; //particle radius = 1mm
    particle.prevPosition.set(0,0,0);
    particle.prevVelocity.set(0,0,0);

    //draw the sphere at the tip of the knife
    //tip of the knife: 0,-1500,0 but make it 0,-1450,0 to see it
    //particle.currentPosition.set(10,10,10);
    //particle.currentPosition.set(cover -> getPointerMat() * cover -> getInvBaseMat()).postMult(knife.lengthInOR); //base units are in m
    //particle.currentPosition.set(cover -> getPointerMat()).postMult(knife.lengthInWC); //in world coordinates, units are mm
    //cout << "in init(), starting position: " << particle.currentPosition.x() << " " << particle.currentPosition.y() << " " << particle.currentPosition.z() << endl; //testing

    particle.mass = 0.05;
    particle.dragModel = cdModel::CD_MOLERUS;
    particle.timeElapsed = double(cover -> frameDuration());
    
    //any arbitrary velocity, change to velocity of knife later
    //particle.currentVelocity = osg::Vec3(10,0,100);

    //function calls to find the values of the fluid dynamics variables
    // particle.findReynoldsNum();
    // particle.findDragCoefficient();
    // particle.findWindForce();
    // particle.findTerminalVelocity();
    
    //*********************************************************************************************************************************DRAWING BLOOD
    //creating a red sphere
    bloodGeode = new osg::Geode;
    
    //matrix controlling the movement of the sphere
    bloodTransform = new osg::MatrixTransform;
    bloodBaseTransform.set(t);
    bloodTransform -> setMatrix(bloodBaseTransform);
    bloodTransform -> addChild(bloodGeode);
    bloodTransform -> setName("bloodTransform");
    
    //create a sphere to model the blood, radius 10, position at the tip of the knife
    bloodSphere = new osg::Sphere(particle.prevPosition, 10);

    bloodShapeDrawable = new osg::ShapeDrawable(bloodSphere);
    bloodShapeDrawable -> setColor(bloodColor);
    bloodGeode -> addDrawable(bloodShapeDrawable);
    bloodGeode -> setName("bloodGeode");

    //********************************RENDERING THE SPHERE IN OPENCOVER************************************
    if(bloodGeode) {
    	bloodStateSet = bloodGeode -> getOrCreateStateSet(); //stateset controls all of the aesthetic properties of the geode
		bloodMaterial = new osg::Material;
		
		//setting the color properties for the sphere
		bloodMaterial -> setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
		bloodMaterial -> setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 0.2f, 1.0f));
		bloodMaterial -> setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 1.0f, 0.2f, 1.0f));
		bloodMaterial -> setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
		bloodMaterial -> setShininess(osg::Material::FRONT_AND_BACK, 25.0);
		bloodStateSet -> setAttributeAndModes(bloodMaterial);
		bloodStateSet -> setNestRenderBins(false);
		
		cover -> getObjectsRoot() -> addChild(bloodTransform);

    	cout << "Hello Blood" << endl;
    }
 
    //*****************************************************************************************************************CREATING THE BLOOD PLUGIN UI MENU
    bloodMenu = new ui::Menu("Blood", this);
    bloodMenu->setText("Blood");

    addBlood = new ui::Action(bloodMenu, "addBlood");
    addBlood->setText("Add Blood");
    addBlood->setCallback([this]() {doAddBlood(); });
    
    cout << "frame duration: " << cover -> frameDuration() << endl; //frame duration: 1.3 seconds
        
    return true;
}

//unit discrepancy(?): do calcs in m for consistency
bool BloodPlugin::update() { 
    osg::Matrix hand = cover -> getPointerMat();
    osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();

    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rot;
    rot.makeRotate(-osg::PI_2, osg::Vec3(0,0,1)); 

    //moving and drawing the knife
    knifeTransform -> setMatrix(rot * handInObjectsRoot);

    //knife.currentPosition = knife.lengthInWC * rot * handInObjectsRoot;
    knife.currentPosition = knife.lengthInOR * rot * handInObjectsRoot;
    
    if(knife.prevPosition.x() == 0) { //will be true on the first time update() is run
        knife.prevPosition = knife.currentPosition;
        particle.currentPosition/*. set(0,0,0) */ = knife.currentPosition; //testing
        //cout << "from init(), particle position is: " << particle.currentPosition << endl; //testing
    }

    if(firstUpdate && particle.currentVelocity.length() != 0) {
        particle.findReynoldsNum();
        particle.findDragCoefficient();
        //particle.findWindForce();
        particle.findTerminalVelocity(); //terminal velocity approx 27 m/s (92 km/h)
        //cout << "particle terminal velocity: " << particle.terminalVelocity << endl;
        firstUpdate = false;
    }
    

    knife.shift = knife.currentPosition - knife.prevPosition;

    knife.prevPosition = knife.currentPosition;

    static double prevKnifeSpeed;
    double knifeAccel;

    static bool elseEntered = false; //testing
    if(particle.onKnife) {
        prevKnifeSpeed = 0;
        double knifeSpeed = knife.shift.length() / (cover -> frameDuration());
        knifeAccel = (knifeSpeed - prevKnifeSpeed) / cover -> frameDuration();
        prevKnifeSpeed = knifeSpeed;

        //if(knifeSpeed != 0) cout << "knife speed: " << knifeSpeed << endl;
        if(knifeAccel != 0) cout << "knife acceleration: " << knifeAccel << endl << endl;
    }

    //?: need a way to limit the position/velocity of the particle, can do terminal velocity/bounded z-position?

    //*************************************************************************************
    if(!particle.onKnife || (particle.currentVelocity.length() > 5 && particle.currentVelocity.length() < 50)) {
        //particle leaves the knife with an initial velocity equal to the knife's velocity
        //cout << "frame duration: " << cover -> frameDuration() << endl;
        
        //preventing the particle from moving backwards
        if(particle.currentVelocity.x() < 0) {
            cout << "changing the direction of x-component of velocity" << endl;
            particle.currentVelocity.x() = abs(particle.currentVelocity.x());
        }
        if(particle.currentVelocity.y() < 0) {
            cout << "changing the direction of y-component of velocity" << endl;
            particle.currentVelocity.y() = abs(particle.currentVelocity.y());
        }

        //checking that velocity < terminal velocity
        if(abs(particle.currentVelocity.x()) < particle.terminalVelocity && 
           abs(particle.currentVelocity.y()) < particle.terminalVelocity && 
           abs(particle.currentVelocity.z()) < particle.terminalVelocity) {

                particle.currentVelocity += particle.gravity * double(cover -> frameDuration());
                
        } else if(abs(particle.currentVelocity.x()) >= particle.terminalVelocity) {
            particle.currentVelocity.set(particle.terminalVelocity, particle.currentVelocity.y(), particle.currentVelocity.z());
            
        } else if(abs(particle.currentVelocity.y()) >= particle.terminalVelocity) {
            particle.currentVelocity.set(particle.currentVelocity.x(), particle.terminalVelocity, particle.currentVelocity.z());
            
        } else if(abs(particle.currentVelocity.z()) >= particle.terminalVelocity) {
            particle.currentVelocity.set(particle.currentVelocity.x(), particle.currentVelocity.y(), -particle.terminalVelocity);
        
        } else {
            particle.currentVelocity.set(particle.terminalVelocity, particle.terminalVelocity, -particle.terminalVelocity);
        }

        particle.currentPosition += particle.currentVelocity * double(cover -> frameDuration());
        particle.onKnife = false;
            
        if(particle.currentVelocity.length() != 0 /* && elseEntered *//* once */) {
            cout << "if-statement particle speed: " << particle.currentVelocity << endl; //testing
            cout << "particle position: " << particle.currentPosition << endl << endl;
            // elseEntered = false;
        }
        
    } else {
        //particle has the same velocity as the knife
        particle.currentVelocity.set(knife.shift / double(cover -> frameDuration()));
        particle.currentPosition.set(/* osg::Vec3(-60,-1300,0) */knife.currentPosition); //move this to a visible position
        if(particle.currentVelocity.length() != 0) {
            cout << "else-statement particle speed: " << particle.currentVelocity << endl << endl; //testing
            elseEntered = true;
        }
    }
    //**************************************************************************************/

    particle.matrix.makeTranslate(particle.currentPosition);
    bloodTransform -> setMatrix(particle.matrix);

    return true;
}

void BloodPlugin::doAddBlood() { //this function causes seg faults
    //create a bunch of blood on the object
    bloodJunks.push_back(new Blood());
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)


    // if(firstUpdate) { //for the first iteration, set the particle velocity to be the same as the knife since you know the particle is on the knife at t = 0
    //     particle.currentVelocity.set(knife.shift / (cover -> frameDuration()));
    //     firstUpdate = false;
    // }

    // osg::Vec3 acceleration;
    // if(particle.onKnife) { //acceleration is only relevant if the particle is on the knife
    //     acceleration = (particle.currentVelocity - particle.prevVelocity)/*  / double(cover -> frameDuration()) */;
    //     particle.prevVelocity = particle.currentVelocity;
    // }

    // if(acceleration.length() != 0) cout << "acceleration: " << acceleration << endl; //testing

    // if((!particle.onKnife || (acceleration.length() > 2000 && acceleration.length() < 10000))) {
    //     particle.currentVelocity.set(particle.gravity * cover -> frameDuration());
    //     particle.onKnife = false;

    // } else {
    //     particle.currentVelocity.set(knife.shift / (cover -> frameDuration())); //travels with same speed as knife

        
    //     //particle.currentPosition += particle.currentVelocity * (cover -> frameDuration());
    //     //cout << "in else block of if-else statement" << endl << endl; //testing

    //     // if(acceleration.length() > 10000) {
    //     //     particle.onKnife = false;
    //     //     //cout << "set particle.onKnife to false in second if-else statement" << endl << endl; //testing
    //     // }
    // }
    
    
    //particle.currentPosition += particle.currentVelocity * cover -> frameDuration();