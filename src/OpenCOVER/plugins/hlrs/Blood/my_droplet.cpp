#include "droplet.h"

using namespace std;

///*********************************************Accessors***************************************/
//    double Blood::getRadius() {
//        return radius;
//    }
//    
//    double Blood::getMass() {
//        return dropletMass;
//    }
//    
//    bool Blood::isOnKnife() {
//        return onKnife;
//    }
//    
//    double Blood::getReynoldsNum() {
//        return ReynoldsNum;
//    }
//    
//    double Blood::getDragCoefficient() {
//        return dragCoeff;
//    }
//    
//    cdModel Blood::getDragModel() {
//        return dragModel;
//    }
//    
//    double Blood::getWindForce() {
//        return windForce;
//    }
//    
//    /********************************************Mutators****************************************/
//    void Blood::setRadius(double _radius) {
//        radius = _radius;
//    }
//    
//    void Blood::setMass(double _mass) {
//        dropletMass = _mass;
//    }
//    
//    void Blood::setOnKnife(bool yesNo) {
//        onKnife = yesNo;
//    }
//    
//    void Blood::setReynoldsNum(double Re) {
//        ReynoldsNum = Re;
//    }
//    
//    void Blood::setDragCoefficient(double cd) {
//        dragCoeff = cd;
//    }
//    
//    void Blood::setDragModel(int model) {
//        if(model == 0) {
//            dragModel = cdModel::CD_STOKES;
//        } else if(model == 1) {
//            dragModel = cdModel::CD_MOLERUS;
//        } else if(model == 2) {
//            dragModel = cdModel::CD_MUSCHELK;
//        } else if(model == 3) {
//            dragModel = cdModel::CD_NONE;
//        }
//    }
//    
//    void Blood::setWindForce(double _windForce) {
//        windForce = _windForce;
//    }
    
    /********************************Determining Reynolds Number********************************/
    /* Reynolds Number (Re): Dimensionless number to predict flow patterns in different fluid flow situations
    * formula: Re = (rho * u * L) / mu
       - rho: density of the fluid (SI units: kg/m^3)
       - u: velocity of the fluid with respect to the object (m/s)
       - L: characteristic linear dimension (m), typically use radius or diameter for circles/spheres
       - mu: dynamic viscosity of the fluid (Pa·s or N·s/m^2 or kg/m·s)

    * onset of turbulent flow: 2.3x10^3 -> 5.0x10^4 for pipe flow, 10^6 for boundary layers
    - reference numbers: Re for blood flow in brain = 1x10^2, Re for blood flow in aorta = 1x10^3  */
    
    void Blood::findReynoldsNum() {
        double Re;
        Re = (RHO_BLOOD * netVelocity(dropletVelocity) * 2 * radius) / DYN_VISC_BLOOD;
        cout << "Re = " << Re << endl;

        if(Re >= REYNOLDS_LIMIT) {
            cout << "Drag modeling behaves correctly until Re = "<< REYNOLDS_LIMIT << " ! Currently Re = " << Re << "\nPropagation may be incorrect!" << endl;
        } else {
            cout << "Below Reynolds Number limit, proceed" << endl; //test
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
    void Blood::findDragCoefficient() {
        double cdLam;
        
        if(dragModel == cdModel::CD_STOKES) {
            cdLam = 24/ReynoldsNum;
            cout << "Cd_lam = " << cdLam << endl;
            
            //???????????????????????????????????cdLam is always a small number because ReynoldsNum is on the order of magnitude of 10^5
            //thus cdLam will almost never be > CD_TURB so function will always return CD_TURB
            
            if(cdLam > CD_TURB) {//idk why you have to do this but they did it in gen.cpp so...
                cout << "cdLam > cdTurb (0.15)" << endl;
                //return cdLam;
                dragCoeff = cdLam;
            } else {
                cout << "cdLam <= cdTurb (0.15)" << endl;
                //return CD_TURB;
                dragCoeff = CD_TURB;
            }
            
        } else if(dragModel == cdModel::CD_MOLERUS) {
            cdLam = 24/ReynoldsNum + 0.4/sqrt(ReynoldsNum) + 0.4;
            cout << "Cd_lam = " << cdLam << endl;
            //return cdLam;
            dragCoeff = cdLam;
            
        } else if(dragModel == cdModel::CD_MUSCHELK) {
            cdLam = 21.5/ReynoldsNum + 6.5/sqrt(ReynoldsNum) + 0.23;
            cout << "Cd_lam = " << cdLam << endl;
            //return cdLam;
            dragCoeff = cdLam;
            
        } else if(dragModel == cdModel::CD_NONE) {
            cdLam = 0.0;
            cout << "Cd_lam = " << cdLam << endl;
            //return cdLam;
            dragCoeff = cdLam;
            
        } else {
            cout << "Cd_lam = 0.47" << endl;
            //return 0.47; //drag coefficient for a smooth sphere with Re = 1x10^5 (from Wikipedia)
            dragCoeff = 0.47;
        }
    }
    
    void Blood::findWindForce() {
        //k = 0.5*densityOfFluid*p->r*p->r*Pi*cwTemp/p->m
        double k = 0.5 * RHO_BLOOD * radius * radius * PI * dragCoeff / dropletMass;
        windForce = k;
    }
    
    void Blood::findTerminalVelocity() {
        //terminal velocity = sqrt((2*m*g)/(dragcoefficient * rho * cross-sectional area))
        double vMax = sqrt((2 * dropletMass * GRAVITY) / (dragCoeff * RHO_BLOOD * crossSectionalArea()));
        terminalVelocity = vMax;
    }
    
//    double velocityWithAirResistance() { //how air resistance affects velocity in x/y/z?
//        //p->velocity -= p->velocity*k*v*timesteps*0.5+gravity*timesteps/2
//        
//        double v_AR = 0.5 * currentVelocity - currentVelocity* windForce * abs(currentVelocity) * deltaTime + 0.5 * GRAVITY * deltaTime;
//        currentVelocity = newVelocity;
//        newVelocity -= v_AR;
//        return newVelocity;
//    }
    
    //from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    void Blood::airResistanceVelocity() {
        //v_x = v_o * cos(theta) * exp(-gravity * time / terminalVelocity)
        //v_z = v_o * sin(theta) * exp(-gravity * time / terminalVelocity) - terminalVelocity * (1 - exp(-gravity * dropletTimeElapsed / terminalVelocity))
        double velocityInX = dropletVelocity.x() * exp((-1) * GRAVITY * dropletTimeElapsed / terminalVelocity());
        double velocityInZ = dropletVelocity.z() * exp((-1) * GRAVITY * dropletTimeElapsed / terminalVelocity()) - terminalVelocity() * (1 - exp((-1) * GRAVITY * dropletTimeElapsed / terminalVelocity()));
        
        //either assign these values to the dropletVelocity member var or make a new osg::Vec3 var
        //haven't decided yet
        dropletVelocity.x() = velocityInX;
        dropletVelocity.z() = velocityInZ;
        
    }
    
    void Blood::determinePosition() {
        //x = v_o * terminalVelocity * cos(theta) / g * (1-exp(-gravity * timeElapsed / terminalVelocity))
        //z = terminalVelocity/gravity * (v_o * sin(theta) + terminalVelocity) * (1-exp(-gravity * timeElapsed / terminalVelocity)) - terminalVelocity * timeElapsed
        
        double positionInX = (dropletVelocity.x() * terminalVelocity()) / (GRAVITY * (1-exp(-1 * GRAVITY * dropletTimeElapsed / terminalVelocity)));
        double positionInZ = (terminalVelocity / GRAVITY) * (dropletVelocity.z() + terminalVelocity) * (1-exp(-1 * GRAVITY * dropletTimeElapsed / terminalVelocity)) - (terminalVelocity * dropletTimeElapsed);
        
        //assign these values to the dropletPosition member varial=ble or return a new osg::Vec3 type
        //haven't decided yet
        dropletPosition.x() = positionInX;
        dropletPosition.z() = positionInZ;
    }
    
    double Blood::maxPosition() {
        //farthest possible distance in the x and z directions (use this for error checking)
        //x = v_o * terminalVelocity * cos(theta) / g
        //z = terminalVelocity/gravity * (v_o * sin(theta) + terminalVelocity) - terminalVelocity * timeElapsed
        
        double maxX = dropletVelocity.x() * terminalVelocity / GRAVITY;
        double maxZ = terminalVelocity / GRAVITY * (dropletVelocity.z() + terminalVelocity) - (terminalVelocity * dropletTimeElapsed);
        
        maxDisplacement.x() = maxX;
        maxDisplacement.z() = maxZ;
    }