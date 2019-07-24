/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Hello OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                             **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cmath>

#include "Blood.h"
#include "BloodPlugin.h"
//#include "globals.h"

using namespace opencover;

Blood::Blood() {
cerr << "Hello Blood" << endl;
    sphere = new coSphere();
    a.set(0, 0, -9.81);

    sphere->setMaxRadius(100);
    sphere->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);    //Doesn't work properly on AMD RADEON 7600M

    BloodPlugin::instance()->bloodNode->addChild(sphere); //causes segmentation fault in function doAddBlood()
    //seg fault reason: BloodPlugin::instance returns inst, inst is initialized to NULL
}

// this is called if the plugin is removed at runtime
Blood::~Blood() {
    BloodPlugin::instance()->bloodNode->removeChild(sphere);
}

void Blood::integrate(float dt, osg::Vec3 vObj) {
    for (int i = 0; i < drops.size(); i++)
    {
        Drop *drop = &drops[i];
        if (drop->sticking)
        {
            osg::Vec3 as = -(drop->v-vObj) * friction*dt;
            drop->v = drop->v + as;
        }
        drop->v = drop->v + a;
        drop->pos = drop->pos + (drop->v * dt);
    }
}

//*************************************************************************************Functions for class Droplet
Droplet::Droplet() {
}

Droplet::~Droplet() {
}

//********************************Determining Reynolds Number********************************
/* Reynolds Number (Re): Dimensionless number to predict flow patterns in different fluid flow situations
* formula: Re = (rho * u * L) / mu
   - rho: density of the fluid (SI units: kg/m^3)
   - u: velocity of the fluid with respect to the object (m/s)
   - L: characteristic linear dimension (m), typically use radius or diameter for circles/spheres
   - mu: dynamic viscosity of the fluid (Pa·s or N·s/m^2 or kg/m·s)

* onset of turbulent flow: 2.3x10^3 -> 5.0x10^4 for pipe flow, 10^6 for boundary layers
- reference numbers: Re for blood flow in brain = 1x10^2, Re for blood flow in aorta = 1x10^3  */

void Droplet::findReynoldsNum() {
    double Re;
    Re = (RHO_BLOOD * pythagoras(currentVelocity) * 2 * radius) / DYN_VISC_BLOOD;
    //cout << "Re = " << Re << endl;

    if(Re >= REYNOLDS_LIMIT) {
        cout << "Drag modeling behaves correctly until Re = "<< REYNOLDS_LIMIT << " ! Currently Re = " << Re << "\nPropagation may be incorrect!" << endl;
    } else {
        //cout << "Below Reynolds Number limit, proceed" << endl; //test
    }
    
//        if(Re >= REYNOLDS_THRESHOLD) { //REYNOLDS_THRESHOLD = 2230 from gen.cpp
//            if(Re >= REYNOLDS_LIMIT) { //REYNOLDS_LIMIT = 170000 from gen.cpp
//                cout << "Drag modeling behaves correctly until Re = "<< REYNOLDS_LIMIT << " ! Currently Re = " << Re << "\nPropagation may be incorrect!" << endl;
//            }
//            return CD_TURB;
//        } else {
//            cout << "Below Reynolds Threshold, proceed" << endl; //test
//        }

    //return Re;
    ReynoldsNum = Re;
}

/*******************Based on the velocity of the particle, different means of calculating the drag will be used****************/
void Droplet::findDragCoefficient() {
    double cdLam;
    
    if(dragModel == cdModel::CD_STOKES) {
        cdLam = 24/ReynoldsNum;
        //cout << "Cd_lam = " << cdLam << endl;
        
        //???????????????????????????????????cdLam is always a small number because ReynoldsNum is on the order of magnitude of 10^5
        //thus cdLam will almost never be > CD_TURB so function will always return CD_TURB
        
        if(cdLam > CD_TURB) {//idk why you have to do this but they did it in gen.cpp so...
            //cout << "cdLam > cdTurb (0.15)" << endl;
            //return cdLam;
            dragCoeff = cdLam;
        } else {
            //cout << "cdLam <= cdTurb (0.15)" << endl;
            //return CD_TURB;
            dragCoeff = CD_TURB;
        }
        
    } else if(dragModel == cdModel::CD_MOLERUS) {
        cdLam = 24/ReynoldsNum + 0.4/sqrt(ReynoldsNum) + 0.4;
        //cout << "Cd_lam = " << cdLam << endl;
        //return cdLam;
        dragCoeff = cdLam;
        
    } else if(dragModel == cdModel::CD_MUSCHELK) {
        cdLam = 21.5/ReynoldsNum + 6.5/sqrt(ReynoldsNum) + 0.23;
        //cout << "Cd_lam = " << cdLam << endl;
        //return cdLam;
        dragCoeff = cdLam;
        
    } else if(dragModel == cdModel::CD_NONE) {
        cdLam = 0.0;
        //cout << "Cd_lam = " << cdLam << endl;
        //return cdLam;
        dragCoeff = cdLam;
        
    } else {
        //cout << "Cd_lam = 0.47" << endl;
        //return 0.47; //drag coefficient for a smooth sphere with Re = 1x10^5 (from Wikipedia)
        dragCoeff = 0.47;
    }
}

void Droplet::findWindForce() {
    //k = 0.5*densityOfFluid*p->r*p->r*Pi*cwTemp/p->m
    double k = 0.5 * RHO_BLOOD * radius * radius * PI * dragCoeff / mass;
    windForce = k;
}

void Droplet::findTerminalVelocity() {
    //terminal velocity = sqrt((2*m*g)/(dragcoefficient * rho * cross-sectional area))
    double vMax = sqrt((2 * mass * GRAVITY) / (dragCoeff * RHO_BLOOD * crossSectionalArea(radius)));
    terminalVelocity = vMax;
}

//http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
osg::Vec3 Droplet::airResistanceVelocity() {
    //v_x = v_o * cos(theta) * exp(-gravity * time / terminalVelocity)
    //v_z = v_o * sin(theta) * exp(-gravity * time / terminalVelocity) - terminalVelocity * (1 - exp(-gravity * timeElapsed / terminalVelocity))
    double velocityInY = currentVelocity.y() * exp((-1) * GRAVITY * timeElapsed / terminalVelocity);
    double velocityInZ = currentVelocity.z() * exp(-1 * GRAVITY * timeElapsed / terminalVelocity) - terminalVelocity * (1 - exp(-1 * GRAVITY * timeElapsed / terminalVelocity));
    
    osg::Vec3 velVec;
    velVec = osg::Vec3(currentVelocity.x(), velocityInY , velocityInZ);
    return velVec;    
}

osg::Vec3 Droplet::determinePosition() {
    //x = v_o * terminalVelocity * cos(theta) / g * (1-exp(-gravity * timeElapsed / terminalVelocity))
    //z = terminalVelocity/gravity * (v_o * sin(theta) + terminalVelocity) * (1-exp(-gravity * timeElapsed / terminalVelocity)) - terminalVelocity * timeElapsed
    
    double positionInX, positionInY, positionInZ;

    osg::Vec3 posVec;

    //positionInX = currentVelocity.x() * timeElapsed; 
    positionInY = (currentVelocity.y() * terminalVelocity) / GRAVITY * (1-exp(-1 * GRAVITY * timeElapsed / terminalVelocity));
    positionInX = currentVelocity.x() * timeElapsed;
    //positionInZ = currentVelocity.z() * timeElapsed; 
    positionInZ = (terminalVelocity / GRAVITY) * (currentVelocity.z() + terminalVelocity) * (1 - exp(-1 * GRAVITY * timeElapsed / terminalVelocity)) - (terminalVelocity * timeElapsed);
    
    //assign these values to the dropletPosition member variable or return a new osg::Vec3 type haven't decided yet
    posVec = osg::Vec3(positionInX, positionInY, positionInZ);
    return posVec;
}

void Droplet::maxPosition() {
    //farthest possible distance in the x and z directions (use this for error checking)
    //x = v_o * terminalVelocity * cos(theta) / g
    //z = terminalVelocity/gravity * (v_o * sin(theta) + terminalVelocity) - terminalVelocity * timeElapsed
    
    double maxX = currentVelocity.x() * terminalVelocity / GRAVITY;
    double maxZ = terminalVelocity / GRAVITY * (currentVelocity.z() + terminalVelocity) - (terminalVelocity * timeElapsed);
    
    maxDisplacement.x() = maxX;
    maxDisplacement.z() = maxZ;
    maxDisplacement.y() = currentPosition.y();
}
    
//*************************************************************************************Functions for class Weapon
Weapon::Weapon() {
}

Weapon::~Weapon() {
}

/*
bool isSlipping() {
    //need to isolate the 4th row of the getPointerMat() matrix to determine the position of the knife
    osg::Matrix knifePos = cover -> getPointerMat();
    osg::Vec4 fourthAxis = osg::Vec4(0,0,0,1);

    osg::Vec4 fourthRow = knifePos.postMult(fourthAxis);
    osg::Vec3 newRefAxis = osg::Vec3(fourthRow.x(), 0,0); //set the new reference axis as a line in the x-direction through the midpoint of the knife
    osg::Vec3 knifeTip = knifePos.postMult(knife.length); //vecor from the origin through the tip of the knife
    
    double hypotenuse = pythaoras(knifeTip.x(), knifeTip.y(), knifeTip.z());
    double cosTheta = knifeTip.z() / hypotenuse;
    double sinThea = hypot(knifeTip.x(), knifeTip.y()) / hypotenuse;

    double a_x = GRAVITY * sinTheta - COEFF_STATIC_FRICTION * GRAVITY * cosTheta;

    double newDropletPosition = 0.5 * a_x * particle.timeElapsed * particle.timeElapsed + particle.currentparticle.currentVelocity * particle.timeElapsed;
    
    
    
    osg::Vec3 knifeTranslation = osg::Vec3(fourthRow.x(), fourthRow.y(), fourthRow.z()); //isolate the translation elements of the 4th row of PointerMat()

    double hypot = pythagoras(knifeTranslation.x(), knifeTranslation.y(), knifeTranslation.z());

    double sinTheta = knifeTranslation.z() / hypot; 
    double cosTheta = knifeTranslation.x() / hypot;

    double a_x = GRAVITY * sinTheta - COEFF_STATIC_FRICTION * GRAVITY * cosTheta;

    //if a_x < 0, static friction force > x-component of gravity
    //if a_x > 0, x-component of gravity > static friction force
    if(a_x < 0) {
        return false;
    } else if(a_x > 0) {
        return true;
    }
}
 */