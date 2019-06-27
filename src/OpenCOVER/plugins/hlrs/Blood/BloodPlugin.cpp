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

using namespace opencover;

/******************************************CARTESIAN COORDINATE SYSTEM FOR OPENCOVER************************************
					y
					y
					y
					y
					y
					y
xxxxxxxxxxxxxxxxxxxxx
					z
				  z 
			    z
		      z
	        z 
          z 
        z
      z
    z
************************************************************************************************************************/

BloodPlugin::BloodPlugin() : ui::Owner("Blood", cover->ui)
{
    fprintf(stderr, "BloodPlugin\n");
}

BloodPlugin *BloodPlugin::inst=nullptr;

// this is called if the plugin is removed at runtime
BloodPlugin::~BloodPlugin()
{
    cover->getObjectsRoot()->removeChild(bloodNode.get());
    for (auto junk = bloodJunks.begin(); junk != bloodJunks.end(); junk++)
    {
        delete *junk;
    }
    bloodJunks.clear();
}

bool BloodPlugin::init() {
    //defines the transformation matrix in x,y,z world coordinates and sets artificial 4th coordinate to 1
    double t[] = {1,0,0,0, //x scaling
                  0,1,0,0, //y scaling
                  0,0,1,0, //z scaling
                  0,0,0,1}; //translation operation in x/y/z direction
		  
    //****************************************************************************************************************************DRAWING THE KNIFE	  
    //loading an external 3D knife into opencover
    //?????????????????????????????????????change the knife dimensions
    osg::Node* knife;    
    knife = coVRFileManager::instance()->loadIcon("knife"); //loads 3d model of a knife into OpenCOVER
    knife -> setName("Knife");
    
    //****************************ANIMATION PATH FOR THE KNIFE TO MOVE IN AN ARC***********************
    //insert(double time, controlPoint)
    osg::ref_ptr<osg::AnimationPath> knifePath = new osg::AnimationPath();

	osg::Quat knifeQuatUp, knifeQuatDown;
    knifeQuatUp.makeRotate(osg::PI_2, osg::Y_AXIS);

	osg::AnimationPath::ControlPoint knifeUp, knifeDown;
	knifeUp.setRotation(knifeQuatUp); //rotation of +180 degrees about the y axis
    
    knifePath -> insert(0.0, knifeUp);
    knifePath -> insert(1.0, knifeDown);
    knifePath -> setLoopMode(osg::AnimationPath::LoopMode::SWING);
    
    osg::ref_ptr<osg::PositionAttitudeTransform> knifePAT = new osg::PositionAttitudeTransform;
    osg::ref_ptr<osg::AnimationPathCallback> tempKnife = new osg::AnimationPathCallback(knifePath);
    knifePAT -> setUpdateCallback(tempKnife);
    knifePAT -> addChild(knife);
    
    /*osg::ref_ptr<osg::MatrixTransform> knifeTransform = new osg::MatrixTransform;
    osg::Matrix knifeBaseTransform;
    knifeBaseTransform.set(t);
    knifeTransform -> setMatrix(knifeBaseTransform);*/
    
    //********************************RENDERING THE KNIFE IN OPENCOVER************************************
    if(knife) { //check if the knife image loaded properly, if so, add it to the graph
        //cover->getObjectsRoot()->addChild(knife); //add the knife to the root of the scene graph
        cover -> getObjectsRoot() -> addChild(knifePAT);
        knifePAT->setName("KnifePAT");
   		cout << "Hello Knife" << endl;
   		cout << "Hello Knife Animation" << endl;
    }
    
    //*********************************************************************************************************************************DRAWING BLOOD
    //creating a red sphere
    osg::ref_ptr<osg::Geode> bloodGeode = new osg::Geode;
    osg::Vec4 bloodColor = osg::Vec4(1., 0., 0., 1.); //RGB: 173, 34, 34 (divide by 255)
    
    //matrix controlling the movement of the sphere
    osg::ref_ptr<osg::MatrixTransform> bloodTransform = new osg::MatrixTransform;
    osg::Matrix bloodBaseTransform;
    bloodBaseTransform.set(t);
    bloodTransform -> setMatrix(bloodBaseTransform);
    
    //create a sphere to model the blood, radius 10, position at the tip of the knife
    osg::ref_ptr<osg::Sphere> bloodSphere = new osg::Sphere(osg::Vec3(-50,0,0), 10);
    osg::ref_ptr<osg::ShapeDrawable> bloodShapeDrawable = new osg::ShapeDrawable(bloodSphere); //add the sphere as a ShapeDrawable
    bloodShapeDrawable -> setColor(bloodColor); //sets the color of the sphere to red (but this doesn't work?)
    bloodGeode -> addDrawable(bloodShapeDrawable); //add the drawable to the Geode
    
    
    //****************************ANIMATION PATH FOR THE SPHERE TO MOVE IN A BOX***********************
    //insert(double time, controlPoint)
    osg::ref_ptr<osg::AnimationPath> bloodPath = new osg::AnimationPath();

	osg::Quat bloodQuatUp;
    bloodQuatUp.makeRotate(osg::PI_2, osg::Y_AXIS);

	osg::AnimationPath::ControlPoint bloodUp, bloodDown;
	bloodUp.setRotation(knifeQuatUp); //rotation of +180 degrees about the y axis
    
    bloodPath -> insert(0.0, bloodUp);
    bloodPath -> insert(1.0, bloodDown);
    bloodPath -> setLoopMode(osg::AnimationPath::LoopMode::SWING);
    
    osg::ref_ptr<osg::PositionAttitudeTransform> bloodPAT = new osg::PositionAttitudeTransform;
    osg::ref_ptr<osg::AnimationPathCallback> tempBlood = new osg::AnimationPathCallback(bloodPath);
    bloodPAT -> setUpdateCallback(tempBlood);
    bloodPAT -> addChild(bloodGeode);
    
    
    //********************************RENDERING THE SPHERE IN OPENCOVER************************************
    if(bloodGeode) {
    	osg::StateSet *stateset = bloodGeode->getOrCreateStateSet(); //stateset controls all of the aesthetic properties of the geode
		osg::Material *material = new osg::Material;
		
		//setting the color properties for the sphere
		material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
		material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 0.2f, 1.0f));
		material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 1.0f, 0.2f, 1.0f));
		material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
		material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
		stateset->setAttributeAndModes(material);
		stateset->setNestRenderBins(false);
		
		//animating the bloodGeode - comment this out if you're going to use AnimationPath instead of matrix transforms
		//cover -> getObjectsRoot() -> addChild(bloodGeode);
		
		//animating the bloodPAT - comment this out if you're going to use matrix transforms instead of AnimationPath
    	cover -> getObjectsRoot() -> addChild(bloodPAT);
        bloodPAT->setName("bloodPAT");
        
    	cout << "Hello Blood" << endl;
    	cout << "Hello Blood Animation" << endl;
    }
 
    //*****************************************************************************************************************CREATING THE BLOOD PLUGIN UI MENU
    bloodMenu = new ui::Menu("Blood", this);
    bloodMenu->setText("Blood");

    addBlood = new ui::Action(bloodMenu, "addBlood");
    addBlood->setText("Add Blood");
    addBlood->setCallback([this]() {doAddBlood(); });
    bloodNode = new osg::Group();
    bloodNode->setName("Blood");
    cover->getObjectsRoot()->addChild(bloodNode.get());
        
    return true;
}

//hardcode the transformation matrix here instead of using the AnimationPath function
bool BloodPlugin::update() { 
	//transformation matrix for the knife movement
	//should be able to use getPointerMat()
	//osg::Matrix knifeNewPosition = coVRPluginSupport::getPointerMat();
	
	//transformation matrix for the blood particle
	//projectile motion, starting from the tip of the knife (assume knife tip is -50 in the x direction)
	/*double b[] = {1,0,0,0
				  0,1,0,0
				  0,0,1,0
				  10,10,10,1}; //change the position
	osg::Matrix bloodNewPosition;
	bloodNewPosition.set(b);*/

    return true;
}

void BloodPlugin::doAddBlood() {
    //create a bunch of blood on the object
    bloodJunks.push_back(new Blood());
}

BloodPlugin * BloodPlugin::instance() {
    return inst;
}

COVERPLUGIN(BloodPlugin)
