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
    particle.radius = 0.001; //radius = 10mm, Droplet class uses m as base unit
    particle.prevPosition.set(0,0,0);

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
    //particle.velocity = osg::Vec3(10,0,100);

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
    
        
    return true;
}

//unit discrepancy(?): do calcs in m for consistency

/*
- why doesn't the particle start/draw at the same spot as the knife (ie: cover -> getPointerMat())?
- particle transitions too quickly from slipping to freefall, not sure if there's even any friction present
 */

bool BloodPlugin::update() { 
    osg::Matrix hand = cover -> getPointerMat();
    osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();

    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rot;
    rot.makeRotate((osg::PI_2 - osg::PI), osg::Vec3(0,0,1)); 

    //moving and drawing the knife
    knifeTransform -> setMatrix(rot * handInObjectsRoot);

    knife.currentPosition = knife.lengthInWC * rot * handInObjectsRoot;
    //knife.currentPosition = knife.lengthInOR * rot * handInObjectsRoot;

    if(knife.prevPosition.x() == 0) { //from init()
        knife.prevPosition = knife.currentPosition;
        particle.currentPosition = knife.currentPosition;
    }

    knife.shift = knife.currentPosition - knife.prevPosition;

        //cerr << knife.currentPosition[0] << "," << knife.currentPosition[1] << "," << knife.currentPosition[2] << endl;
    knife.prevPosition = knife.currentPosition;

    double knifeSpeed = knife.shift.length() / (cover -> frameDuration());
    //cout << "knife speed: " << knifeSpeed << endl;

    //*************************************************************************************
    // if(knifeSpeed > 5 || !particle.onKnife) {
    //     //particle leaves the knife with an initial velocity equal to the knife's velocity
    //     particle.velocity.set(knife.shift / (cover -> frameDuration()));
    //     particle.onKnife = false;
    // } else {
    //     //particle has the same velocity as the knife
    //     particle.velocity.set(0,0,0);
    //     particle.currentPosition.set(knife.currentPosition);
    // }
    // particle.velocity += particle.gravity * float(cover -> frameDuration());
    // particle.currentPosition += particle.velocity * (cover -> frameDuration());
    //**************************************************************************************/

    
    if(firstUpdate) { //for the first iteration, set the particle velocity to be the same as the knife since you know the particle is on the knife at t = 0
        particle.velocity.set(knife.shift / (cover -> frameDuration()));
        firstUpdate = false;
    }
    
    //double acceleration = isSliding(knife.shift);
    static osg::Vec3 oldVelocity(0,0,0);
    osg::Vec3 acceleration = oldVelocity - particle.velocity;
    oldVelocity = particle.velocity;

    if(acceleration.length() != 0) cout << "acceleration: " << acceleration.length() << endl << endl; //testing

    //if((acceleration.length() > 2 && acceleration.length() < 10) || !particle.onKnife) { //assume the particle left the knife
    if((acceleration.length() > 2000 && acceleration.length() < 10000) /* || !particle.onKnife */) {
        //???????????????????????no downwards acceleration from gravity??????????????????????????????????
        particle.currentPosition += (particle.gravity * 0.5 * pow(cover -> frameDuration(), 2.0)) + (particle.velocity * cover -> frameDuration());
        //particle.onKnife = false;

        // if(!particle.onKnife) { //particle has left the knife, particle is in freefall
        //     //x =  1/2gt^2 + vt
        //     particle.currentPosition = particle.gravity * 0.5 * (cover -> frameDuration() * cover -> frameDuration) + particle.velocity * cover -> frameDuration;
        // } else {
        //     //velocity = relative motion of particle wrt knife + velocity of knife
        //     particle.velocity.set((acceleration * (cover -> frameDuration())) + (knife.shift / (cover -> frameDuration()));
        //     particle.currentPosition += particle.velocity / (cover -> frameDuration());
        // }

    } else { //particle is still on the knife
        particle.velocity.set(knife.shift / (cover -> frameDuration())); //travels with same speed as knife
        particle.currentPosition += particle.velocity * (cover -> frameDuration());
    }

    particle.matrix.makeTranslate(particle.currentPosition);
    bloodTransform -> setMatrix(particle.matrix);


    return true;
}

double BloodPlugin::isSliding(osg::Vec3 deltaPos) {
    // double hypotenuse = signedLength(deltaPos);
    // if(hypotenuse < 0) { //current - prev < 0: current position is below previous position
    //     //assume that if the position is lower then friction force has been overcome
        
    // } else { //current - prev > 0, current position is above previous postion

    // }
    double hypotenuse = deltaPos.length();
    double sinTheta = deltaPos.z() / hypotenuse; //doesnt work since this value is always 0
    double cosTheta = deltaPos.y() / hypotenuse;

    double a_y = (GRAVITY * sinTheta) - (COEFF_STATIC_FRICTION * GRAVITY * cosTheta);

    if(a_y > 0) {
        double pos = (0.5 * a_y * particle.timeElapsed * particle.timeElapsed)/*  + (particle.velocity.y() * particle.timeElapsed) */;

        if(pos > knife.lengthInOR.length()) {
            particle.onKnife = false;
        } else {
            particle.onKnife = true;
        }
    }

    return a_y;
}

void BloodPlugin::doAddBlood() { //seg faults
    //create a bunch of blood on the object
    bloodJunks.push_back(new Blood());
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)