/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BLOOD_H
#define BLOOD_H
/****************************************************************************\ 
 **                                                            (C)2018 HLRS  **
 **                                                                          **
 ** Description: some blood drops                                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** August 2018  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <PluginUtil/coSphere.h>

#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Material>

#include <vector>

#include "globals.h"

//************************************************************************************************Class Drop
class Drop
{
public:
    osg::Vec3 pos;
    osg::Vec3 v;
    float mass;
    float d;
    double getCW() { return 0.15; };
    bool sticking=true;
};

//************************************************************************************************Class Blood
class Blood 
{
    const double densityOfAir = 1.18;
    const double densityOfBlood = 1055.0;
    const double friction = 0.1;
public:
    Blood();
    ~Blood();
    void integrate(float dt, osg::Vec3 vObj);
    std::vector<Drop> drops;
    opencover::coSphere *sphere;
    osg::Vec3 a;
};

//************************************************************************************************Class Droplet
class Droplet{
public:
	Droplet(osg::Vec4 color = osg::Vec4(1,0,0,1)); //default color is red
	~Droplet();

    double timeElapsed; //time between frames
    bool onKnife = true;
    bool firstUpdate = true;
    bool onFloor = false;
    double mass; //units: kg
    
    //kinematics data
    double radius;             //units: m
    osg::Vec3 currentVelocity; //units: m/s, (v_x,v_y,v_z)
    osg::Vec3 prevVelocity;
    osg::Vec3 acceleration;
    osg::Vec3 disp;
    osg::Vec3 currentPosition; //units: m (pos_x, pos_y, pos_z)
    osg::Vec3 prevPosition;
    osg::Vec3 gravity = osg::Vec3(0,0,-9.81);

    osg::Matrix matrix;

    //member variables for displaying blood
    osg::ref_ptr<osg::Geode> bloodGeode;
	osg::Vec4 bloodColor;
	osg::ref_ptr<osg::MatrixTransform> bloodTransform;
	osg::Matrix bloodBaseTransform;
	osg::ref_ptr<osg::Sphere> bloodSphere;
	osg::ref_ptr<osg::ShapeDrawable> bloodShapeDrawable;
	osg::ref_ptr<osg::StateSet> bloodStateSet;
	osg::ref_ptr<osg::Material> bloodMaterial;
    
    osg::Vec3 maxDisplacement; //the furthest distance the droplet can travel before hitting the ground again, excluding trivial solutions
    
    //drag data
    double ReynoldsNum; //dimensionless
    cdModel dragModel;  //CD is an enum type, can be either Stokes/Molerus/Muschelk/None   
    double dragCoeff;   //laminar flow drag coefficient
    double windForce;   //air resistance
    double terminalVelocity;
   
    void findReynoldsNum();
    void findDragCoefficient();
    void findWindForce();
    void findTerminalVelocity();
    osg::Vec3 airResistanceVelocity();
    osg::Vec3 determinePosition();
    void maxPosition();
};

//***************************************************************************************************Class Weapon
class Weapon {
public:
    Weapon();
    ~Weapon();

    double currentSpeed;
    double prevSpeed = 0;
    double accel;

    osg::Node* knifePtr;
    osg::Vec3 a;

    double timeElapsed;
    double mass;
    osg::Vec3 currentVelocity;
    osg::Vec3 prevVelocity;
    osg::Vec3 acceleration;
    
    osg::Vec3 prevPosition;
    osg::Vec3 currentPosition;
    osg::Vec3 shift;
    
    osg::Vec3 start;
    osg::Vec3 end;

    osg::Vec3 tip = osg::Vec3(0,50,-5); //length of the knife in object's root coordinates (m)
    osg::Vec3 hilt = osg::Vec3(1,1,1);
};

#endif
