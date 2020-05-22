
#ifndef DROPLETDATA_H
#define DROPLETDATA_H

#include <iostream>
#include <cmath>
#include "globalConstants.h"
#include "mathFunctions.h"

using namespace std;

class Blood {
private:
    //tracking the movement of the particle
    osg::ref_ptr<osg::MatrixTransform> dropletTransform; //Matrix transform to model the droplet's motion
    osg::ref_ptr<osg::Matrix> dropletMatrix; 
public:
    double dropletTimeElapsed; //time between 2 successive readings, units: s
    
    //kinematics data
    double radius; //units: m
    osg::Vec3 dropletVelocity; //units: m/s, (v_x,v_y,v_z)
    osg::Vec3 dropletPosition; //units: m (pos_x, pos_y, pos_z)
    osg::Vec3 maxDisplacement; //the furthest distance the droplet can travel before hitting the ground again, excluding trivial solutions
    double dropletMass; //units: kg
    
    //mechanics data
    double ReynoldsNum; //dimensionless
    cdModel dragModel; //CD is an enum type, can be either Stokes/Molerus/Muschelk/None   
    double dragCoeff; //laminar flow drag coefficient
    double windForce; //air resistance
    double terminalVelocity;
    
    Blood();
    ~Blood();
    
//    /*********************************************Accessors***************************************/
//    double getRadius();
//    
//    double getMass();
//       
//    double getReynoldsNum();
//    
//    double getDragCoefficient();
//    
//    cdModel getDragModel();
//    
//    double getWindForce();
//    
//    /********************************************Mutators****************************************/
//    void setRadius(double _radius);
//    
//    void setMass(double _mass);
//       
//    void setReynoldsNum(double Re);
//    
//    void setDragCoefficient(double cd);
//    
//    void setDragModel(int model);
//    
//    void setWindForce(double _windForce);
//    
    /********************************Determining Reynolds Number********************************/
    
    void findReynoldsNum();
    
    /*******************Based on the velocity of the particle, different means of calculating the drag will be used****************/
    void findDragCoefficient();
    
    void findWindForce();
    
    void findTerminalVelocity();

    void airResistanceVelocity();
    
    void determinePosition();
    
    double maxPosition();

};

#endif /* DROPLETDATA_H */