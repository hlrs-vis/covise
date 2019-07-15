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
#include <cover/coVRPlugin.h>
#include <PluginUtil/coSphere.h>
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
	Droplet();
	~Droplet();
    double timeElapsed; //time between 2 successive readings, units: s
    bool onKnife = true;;
    
    //kinematics data
    double radius; //units: m
    osg::Vec3 velocity; //units: m/s, (v_x,v_y,v_z)
    osg::Vec3 currentPosition; //units: m (pos_x, pos_y, pos_z)
    osg::Vec3 prevPosition;
    osg::Vec3 gravity = osg::Vec3(0,0,-9.81);

    osg::Matrix matrix;
    
    double a_x;
    double xPosition;

    osg::Vec3 maxDisplacement; //the furthest distance the droplet can travel before hitting the ground again, excluding trivial solutions
    double mass; //units: kg
    
    //drag data
    double ReynoldsNum; //dimensionless
    cdModel dragModel; //CD is an enum type, can be either Stokes/Molerus/Muschelk/None   
    double dragCoeff; //laminar flow drag coefficient
    double windForce; //air resistance
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

    osg::Node* knifePtr;
    osg::Vec3 a;

    double timeElapsed;
    osg::Vec3 velocity;
    osg::Vec3 prevPosition;
    osg::Vec3 currentPosition;
    osg::Vec3 shift;
    osg::Vec3 lengthInWC = osg::Vec3(0,200,0); //length of the knife in world coordinates (mm)
    osg::Vec3 lengthInOR = osg::Vec3(0,0.2,0); //length of the knife in object's root coordinates (m)
    osg::Vec3 tip = osg::Vec3(-60,-1450,0); //knife tip is at 0,-1450,0
    osg::Vec3 tipInmm = osg::Vec3(-0.6, -14.5, 0);
};

#endif

/* */