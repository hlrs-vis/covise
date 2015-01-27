/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __EINSPUR_DYNAMIK_H
#define __EINSPUR_DYNAMIK_H

#include "Vehicle.h"
#include "VehicleDynamics.h"

class EinspurDynamikState
{
public:
    double x;
    double y;
    double kappa;
    double v;
    double alpha;

    double u;

    double epsilon;
    double dEpsilon;

    double zeta;
    double dZeta;

    EinspurDynamikState operator+(const EinspurDynamikState &state) const;
    EinspurDynamikState operator-(const EinspurDynamikState &state) const;
    EinspurDynamikState &operator=(const EinspurDynamikState &state);
    EinspurDynamikState operator*(const double scalar) const;
};

class EinspurDynamikDState
{
public:
    double dX;
    double dY;
    double dKappa;
    double dV;
    double dAlpha;

    double dU;

    double dEpsilon;
    double ddEpsilon;

    double dZeta;
    double ddZeta;

    EinspurDynamikDState operator+(const EinspurDynamikDState &state) const;
    EinspurDynamikDState &operator=(const EinspurDynamikDState &state);

    EinspurDynamikState operator*(const double scalar) const;

    //EinspurDynamikDState operator*(const double scalar) const;
};

class EinspurDynamik : public VehicleDynamics
{
public:
    EinspurDynamik();

    double getAlpha();
    double getX();
    double getY();
    double getVelocity();
    double getAcceleration();
    double getEpsilon();
    double getDEpsilon();
    double getDDEpsilon();
    double getZeta();
    double getDZeta();
    double getDDZeta();
    double getEngineSpeed();
    const osg::Matrix &getVehicleTransformation();
    virtual void setVehicleTransformation(const osg::Matrix &);
    virtual void setRoadType(int);

    void move(VrmlNodeVehicle *vehicle);

    void resetState();

private:
    void setFactors();
    void setInitialState(double x, double y, double v, double alpha);

    void integrate(double h, double x, double y, double alpha);

    EinspurDynamikDState dFunction(EinspurDynamikState y);

    double norm(const EinspurDynamikState &state);

    double beta; // Wheel angle
    double gamma; // Angle acceleration vector and roll axis (Mass point)
    double gammav; // Angle acceleration vector and roll axis (Middle of Vehicle)

    double a; // Acceleration

    double s; // Distance velocity pole (Momentanpol) to rear axes
    double r; // Distance velocity pole (Momentanpol) to vehicle mass point
    double rv; // Distance velocity pole (Momentanpol) to vehicle middle point
    double q; // Distance velocity pole (Momentanpol) to front axes

    double m; // Vehicle mass
    double I; // Vehicle inertia
    double lv; // Vehicle distance front to mass point
    double lh; // Vehicle distance rear to mass point

    double roadFrictionForce;

    double b;
    double h;
    double Aw;
    double cw;

    double Iw;
    double hs;
    double kw;
    double dw;

    double In;
    double kn;
    double dn;

    double wr; // Wheel radius

    double Im; // Motor moment of inertia
    double ach; // Motor characteristic parameters
    double bch;
    double cch;
    double ul; // Engine speed limit
    double ui; // Engine idle speed
    double accPedalIdle;

    double iag; // Axle gearbox
    double i[8]; // Vehicle gearbox

    double cc;
    double kc;

    double cm;
    double km;

    EinspurDynamikState yNull; // Initial value
    EinspurDynamikState yOne; // Integration result
    EinspurDynamikDState dYOne; // Integration dState

    double errorcontrol;
    int num_intsteps;

    int stepcount;

    osg::Matrix chassisTrans;
    osg::Matrix bodyTrans;

    double accPedal;
    double brakePedal;
    double clutchPedal;
    int gear;
    double steerWheelAngle;
};

inline double EinspurDynamik::getAlpha()
{
    return yOne.alpha;
}

inline double EinspurDynamik::getX()
{
    return yOne.x;
}

inline double EinspurDynamik::getY()
{
    return yOne.y;
}

inline double EinspurDynamik::getVelocity()
{
    return yOne.v;
}
inline double EinspurDynamik::getAcceleration()
{
    return a;
}

inline double EinspurDynamik::getEpsilon()
{
    return yOne.epsilon;
}
inline double EinspurDynamik::getDEpsilon()
{
    return yOne.dEpsilon;
}
inline double EinspurDynamik::getDDEpsilon()
{
    return dYOne.ddEpsilon;
}

inline double EinspurDynamik::getZeta()
{
    return yOne.zeta;
}
inline double EinspurDynamik::getDZeta()
{
    return yOne.dZeta;
}
inline double EinspurDynamik::getDDZeta()
{
    return dYOne.ddZeta;
}

inline double EinspurDynamik::getEngineSpeed()
{
    return yOne.u / (2 * M_PI);
}

inline const osg::Matrix &EinspurDynamik::getVehicleTransformation()
{
    return chassisTrans;
}

#endif
