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
    knife.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).postMult(knife.length);
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
    particle.prevPosition = osg::Vec3(0,0,0);

    //draw the sphere at the tip of the knife
    //tip of the knife: 0,-1500,0 but make it 0,-1450,0 to see it
    particle.currentPosition = (cover -> getPointerMat() * cover -> getInvBaseMat()).preMult(osg::Vec3(0,-1450,0)); //base units are in m

    particle.mass = 0.01;
    particle.dragModel = cdModel::CD_MOLERUS;
    particle.timeElapsed = cover -> frameDuration();
    
    //any arbitrary velocity, change to velocity of knife later
    particle.velocity = osg::Vec3(10,0,100);

    //function calls to find the values of the fluid dynamics variables
    particle.findReynoldsNum();
    particle.findDragCoefficient();
    particle.findWindForce();
    particle.findTerminalVelocity();
    particle.maxPosition();
    
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
    bloodSphere = new osg::Sphere(particle.currentPosition, 10);
    cout << "in init(), starting position: " << particle.currentPosition.x() << " " << particle.currentPosition.y() << " " << particle.currentPosition.z() << endl; //testing

    bloodShapeDrawable = new osg::ShapeDrawable(bloodSphere);
    bloodShapeDrawable -> setColor(bloodColor);
    bloodGeode -> addDrawable(bloodShapeDrawable);
    bloodGeode -> setName("bloodGeode");

    // //testing shit
    // osg::ref_ptr<osg::Geode> testGeode = new osg::Geode;
    // osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
    // osg::Matrix testM /* = cover -> getPointerMat() * cover -> getInvBaseMat()*/;
    // // osg::Matrix temp;
    // // temp.makeTranslate(osg::Vec3(-60,-1450,0));
    // // testM *= temp;
    // testM.set(t);
    // mt -> setMatrix(testM);
    // mt -> addChild(testGeode);
    // mt -> setName("testing_stuff");
    // osg::ref_ptr<osg::Sphere> testSphere = new osg::Sphere(/*( cover -> getPointerMat() * cover -> getInvBaseMat()).postMult*/(osg::Vec3(-60,-1450,0)), 10);
    // //knife tip is at approx (-60,-1450,0)
    // osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(testSphere);
    // testGeode -> addDrawable(sd);
    // testGeode -> setName("test");
    // cover -> getObjectsRoot() -> addChild(mt);
    // cout << "Hello test sphere" << endl;
    
    
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

//hardcode the transformation matrix here instead of using the AnimationPath function
//unit discrepancy(?): do calcs in m for consistency
bool BloodPlugin::update() { 
    osg::Matrix hand = cover -> getPointerMat(); //pointerMat is the location of the mouse
    osg::Matrix handInObjectsRoot = cover -> getPointerMat() * cover -> getInvBaseMat();
    
    //rotate clockwise 90 degrees about z-axis
    osg::Matrix rot;
    rot.makeRotate((osg::PI_2 - osg::PI), osg::Vec3(0,0,1)); 

    //moving and drawing the knife
    knifeTransform -> setMatrix(rot * handInObjectsRoot);

    knife.prevPosition = knife.currentPosition;
    knife.currentPosition = handInObjectsRoot.postMult(osg::Vec3(0,-1450,0)); //should be hand or handInObjectsRoot?

    osg::Vec3 knifeShift = knife.currentPosition - knife.prevPosition;
    knife.velocity = (knifeShift / knife.timeElapsed);

    //assume for now the droplet is on the knife which is more or less what is currently happening
    particle.velocity = knife.velocity;
    particle.currentPosition = knife.currentPosition;

    cout << "particle velocity: " << particle.velocity.x() << " " << particle.velocity.y() << " " << particle.velocity.z() << endl;
    cout << "particle position: " << particle.currentPosition.x() << " " << particle.currentPosition.y() << " " << particle.currentPosition.z() << endl << endl;

    //moving and drawing the particle
    (particle.matrix).makeTranslate(particle.currentPosition);
    bloodBaseTransform *= particle.matrix;
    bloodTransform -> setMatrix(bloodBaseTransform);
    return true;
}

/*
To do
- why does the knife draw in the lower right corner of opencover? 
- figure out why the thing won't animate properly if I try to use airResistanceVelocity() and determinePosition() member functions
- fix the isSliding function but this depends on me actually finding the location of the tip of the knife
- not actually sure if the projectile motion based on the knife velocity is rendering correctly so check that
 */

double BloodPlugin::isSliding() {
    //need to isolate the 4th row of the getPointerMat() matrix to determine the position of the knife
    osg::Matrix knifePos = cover -> getPointerMat() * cover -> getInvBaseMat();
    osg::Vec3 knifeTip = knifePos.postMult(knife.length); //vecor from the origin through the tip of the knife

    //necessary?
    osg::Vec4 fourthAxis = osg::Vec4(0,0,0,1);
    osg::Vec4 fourthRow = knifePos.postMult(fourthAxis);
    osg::Vec3 newRefAxis = osg::Vec3(fourthRow.x(), 0,0); //set the new reference axis as a line in the x-direction through the midpoint of the knife
    
    double hypotenuse = knife.tip.length();
    double sinTheta = knife.tip.z() / hypotenuse;
    double cosTheta = hypot(knife.tip.x(), knife.tip.y()) / hypotenuse;

    double a = (GRAVITY * sinTheta) - (COEFF_STATIC_FRICTION * GRAVITY * cosTheta);
    double pos = (0.5 * a * particle.timeElapsed * particle.timeElapsed) + (particle.velocity.x() * particle.timeElapsed);
    
    if(pos > knife.length.length()) {
        particle.onKnife = false;
    } else { 
        particle.onKnife = true;
    }
    if(a > 0) {
        return a;
    } else {
        return 0.0;
    }
   
}

void BloodPlugin::doAddBlood() { //seg faults
    //create a bunch of blood on the object
    bloodJunks.push_back(new Blood());
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)

//when simulating multiple particles and using a vector of particles, remove a particle from the beginning of the vector
    // if(isSliding()) { //frictional force has been overcome
    //     if(!particle.onKnife) { 
    //         //if particle left the knife, then do air resistance calculations
    //         particle.airResistanceVelocity();
    //         particle.prevPosition = particle.currentPosition;
    //         particle.determinePosition();
    //     } else {
    //         particle.velocity = osg::Vec3(particle.a_x * particle.timeElapsed, particle.velocity.y(), particle.velocity.z()); //v and a in x is same as knife
    //         particle.prevPosition = particle.currentPosition;
    //         particle.currentPosition = osg::Vec3(particle.xPosition, particle.prevPosition.y(), particle.prevPosition.z());
    //     }
    // } else { //friction force hasn't been overcome yet, the particle moves as though it is a part of the knife
    //     particle.velocity = knife.velocity;
    //     particle.prevPosition = particle.currentPosition;
    //     particle.currentPosition = knife.currentPosition;
    // }

    // double accel = isSliding(); //??????????????????????????????????????????????????????????????????????i think this function is messed up but thats a tomorrow problem
    // //isSliding is a function which returns the acceleration of the particle if it is sliding on the knife
    // if(accel > 0) {
    //     if(particle.onKnife) { //if the particle is still on the knife, particle.velocity = knife.velocity + particle's relative velocity
    //         particle.velocity.y() = knife.velocity.y() + accel * particle.timeElapsed; //assuming sliding happens in the y-direction
    //         particle.velocity.z() = knife.velocity.z();
    //         particle.velocity.x() = knife.velocity.x();
    //         if(once) cout << "particle.onKnife == TRUE" << endl;
    //     } else { //particle has left the knife, particle.velocity = velocity in air
    //         particle.velocity = particle.airResistanceVelocity();
    //         if(once) cout << "particle.onKnife == FALSE" << endl;
    //     }
    // } else { //the frictional force has not been overcome, particle.velocity = knife.velocity
    //     particle.velocity = knife.velocity;
    //     //if(once) cout << "accel < 0" << endl;
    // }

    // particle.currentPosition = particle.velocity * particle.timeElapsed;

    // //testing
    // // cout << "particle velocity: " << particle.velocity.x() << " " << particle.velocity.y() << " " << particle.velocity.z() << endl;
    // cout << "particle position: " << particle.currentPosition.x() << " " << particle.currentPosition.y() << " " << particle.currentPosition.z() << endl << endl;
    
    // // if(knifeShift.length() == 0) {
    // //     particle.currentPosition = particle.prevPosition;
    // // } else {
   
    // //}
    
    // // particle.prevPosition = particle.currentPosition; //currentPosition in init() initialized to the same thing as handInObjectsRoot
    // // particle.velocity = knife.velocity; //assuming for now the particle is on the knife, it travels with the same velocity as the knife
    // // //particle.velocity += particle.gravity * particle.timeElapsed;//this causes it to not show up on the screen
    // // particle.currentPosition += particle.velocity * particle.timeElapsed; //can't add acceleration otherwise it doesn't show on screen?
    