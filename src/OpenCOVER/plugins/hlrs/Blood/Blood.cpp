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
#include "globals.h"

using namespace opencover;

Blood::Blood() {
cerr << "Hello Blood" << endl;
    sphere = new coSphere();
    a.set(0, 0, -9.81);

    sphere->setMaxRadius(100);
    sphere->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);    //Doesn't work properly on AMD RADEON 7600M

    BloodPlugin::instance()->bloodGeode->addDrawable(sphere);
}

// this is called if the plugin is removed at runtime
Blood::~Blood() {
    BloodPlugin::instance()->bloodGeode->removeDrawable(sphere);
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
//defines the transformation matrix in x,y,z world coordinates and sets artificial 4th coordinate to 1
double d[] = {1.,0.,0.,0., //x scaling
              0.,1.,0.,0., //y scaling
              0.,0.,1.,0., //z scaling
              1.,1.,1.,1.}; //translation operation in x/y/z direction

Droplet::Droplet(osg::Vec4 color) {
    radius = 0.001;
    prevPosition.set(0,0,0);
    prevVelocity.set(0,0,0);
    acceleration.set(0,0,0);

    mass = 0.5;
    dragModel = cdModel::CD_MOLERUS;
    timeElapsed = double(cover -> frameDuration());

    bloodGeode = new osg::Geode;
    
    //matrix controlling the movement of the sphere
    bloodTransform = new osg::MatrixTransform;
    bloodBaseTransform.set(d);
    bloodTransform -> setMatrix(bloodBaseTransform);
    bloodTransform -> addChild(bloodGeode);

    string transformName = "bloodTransform"/*  + std::to_string(id) */;
    bloodTransform -> setName(transformName); //later update name to include the particle number    

    //create a sphere to model the blood
    bloodSphere = new osg::Sphere(prevPosition, 0.01);

    bloodShapeDrawable = new osg::ShapeDrawable(bloodSphere);
    bloodColor = color;
    bloodShapeDrawable -> setColor(bloodColor);
    bloodGeode -> addDrawable(bloodShapeDrawable);

    string geodeName = "bloodGeode"/*  + std::to_string(id) */;
    bloodGeode -> setName(geodeName); //later update name to include the particle number

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
        
        //moved to BloodPlugin.cpp
        // cover -> getObjectsRoot() -> addChild(bloodTransform);
    	// cout << "Hello Blood" << endl;
    }
    
	bloodGeode->setNodeMask((bloodGeode->getNodeMask() & ~Isect::Update )& (~Isect::Intersection));
}

Droplet::~Droplet() {
}

//********************************Determining Reynolds Number********************************
/* Reynolds Number (Re): Dimensionless number to predict flow patterns in different fluid flow situations
* formula: Re = (rho * u * L) / mu
   - rho: density of the fluid (SI units: kg/m^3)
   - u: velocity of the fluid with respect to the object (m/s)
   - L: characteristic linear dimension (m), typically use radius or diameter for circles/spheres
   - mu: dynamic viscosity of the fluid (PaÂ·s or NÂ·s/m^2 or kg/mÂ·s)
*/

void Droplet::findReynoldsNum() {
    double Re;
    Re = (RHO_BLOOD * pythagoras(currentVelocity) * 2 * radius) / DYN_VISC_BLOOD;

    if(Re >= REYNOLDS_LIMIT) {
        cout << "Drag modeling behaves correctly until Re = "<< REYNOLDS_LIMIT << " ! Currently Re = " << Re << "\nPropagation may be incorrect!" << endl;
    }

    ReynoldsNum = Re;
}

/*******************Based on the velocity of the particle, different means of calculating the drag will be used****************/
void Droplet::findDragCoefficient() {
    double cdLam;
    
    if(dragModel == cdModel::CD_STOKES) {
        cdLam = 24/ReynoldsNum;
        if(cdLam > CD_TURB) {
            dragCoeff = cdLam;
        } else {
            dragCoeff = CD_TURB;
        }
        
    } else if(dragModel == cdModel::CD_MOLERUS) {
        cdLam = 24/ReynoldsNum + 0.4/sqrt(ReynoldsNum) + 0.4;
        dragCoeff = cdLam;
        
    } else if(dragModel == cdModel::CD_MUSCHELK) {
        cdLam = 21.5/ReynoldsNum + 6.5/sqrt(ReynoldsNum) + 0.23;
        dragCoeff = cdLam;
        
    } else if(dragModel == cdModel::CD_NONE) {
        cdLam = 0.0;
        dragCoeff = cdLam;
        
    } else {
        dragCoeff = 0.47;
    }
}

void Droplet::findWindForce() {
    //k = 0.5*densityOfFluid*radius^2*Pi*cwTemp/mass
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

    positionInY = (currentVelocity.y() * terminalVelocity) / GRAVITY * (1-exp(-1 * GRAVITY * timeElapsed / terminalVelocity));
    positionInX = currentVelocity.x() * timeElapsed;
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
