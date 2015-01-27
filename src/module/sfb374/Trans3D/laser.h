/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          laser.h  -  laser settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __LASER_H_

#define __LASER_H_

#include <string>

#include "classext.h"

// ***************************************************************************
// class for laser movement
// ***************************************************************************

class TMoveVect
{
public:
    TMoveVect()
    {
        ndTime = ndVelocity = ndRadius = Angle = 0;
        ptBegin = ptEnd = 0;
        iType = linear;
    }

    bool operator==(const TMoveVect &tm) const
    {
        return (ndTime == tm.ndTime);
    }

    void Save(int);
    void Read(int);

    int iType; // motion type
    enum Types
    {
        linear = 0,
        trepanning = 1
    };
    prec ndTime; // time
    prec ndVelocity; // velocity
    prec ndRadius; // radius
    prec Angle; // initial angle
    TPoint ptBegin; // starting point
    TPoint ptEnd; // end or center point
};

inline ostream &operator<<(ostream &ps, const TMoveVect &src)
{
    ps << src.ndTime << '\t' << src.ndVelocity << '\t'
       << src.ndRadius << '\t' << src.Angle << '\t'
       << src.ptBegin << '\t' << src.ptEnd << endl;
    return ps;
}

inline istream &operator>>(istream &ps, TMoveVect &src)
{
    ps >> src.ndTime >> src.ndVelocity >> src.ndRadius
        >> src.Angle >> src.ptBegin >> src.ptEnd;
    ps.get();
    return ps;
}

typedef TStorage<TMoveVect> TMoveArray;

// ***************************************************************************
// laser settings
// ***************************************************************************

class TLaser
{
public:
    TLaser()
    {
        Reset();
    }
    TLaser(const TLaser &);

    void Save(int); // save settings
    void Read(int, float vers = 1.0); // read settings
    void Reset(); // set to default
    void Update(); // update dependent variables
    TLaser GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form

    // read shape file
    void ReadShape(prec *pRad = 0, prec *pInt = 0);
    // read temporal file
    void ReadPulse(prec *pLen = 0, prec *pFlu = 0,
                   prec *pDuty = 0);
    void ReadMove(); // read movement file

    prec GetPower(prec); // get power (temporal)
    // get flux and direction
    prec GetFlux(const TPoint3D &, TPoint3D &);
    TPoint GetMovement(prec); // get rel. movement
    TPoint3D GetPosition(prec); // get focus position

    void CalcAbsorbedHeat(); // calculate laser absorption in solid

    string ProfileName; // name of used profile
    bool bChanged; // true if changed

    prec RefIntensity; // average intensity I0 [W/m2] = F/tp
    prec Fluence; // fluence F[J/m^2]

    prec Radius; // radius w[m]
    prec Divergence; // divergence theta [rad]
    prec ndWavelength; // wavelength lambda (scale w)

    string ShapeName; // user file for intensity distribution
    Interpolation2D ndGeomShape; // intensity distribution (x,y) in focus
    // (scaled by intensity in center)
    prec Mode; // lasermode: 0..1 = TEM01..TEM00
    // 2,3,4 = Top-Hat round, sqare, rhomb
    // 5 = user defined
    enum Lasermode
    {
        TEM01 = 0,
        TEM00 = 1,
        THround = 2,
        THsquare = 3,
        THrhomb = 4,
        user_mode = 5
    };
    prec ndHeight; // height for rhomb-mode

    string PulseName; // user file for temporal intensity
    rinterpol ndPulseShape; // temporal shape (scaled by I0)
    int iPulseForm; // temporal pulseform: 0 = cw,
    // 1 = rectangular, 2 = user defined
    enum Pulseform
    {
        cw = 0,
        rectangular = 1,
        user_form = 2
    };
    prec Pulselength; // pulselength [s]
    prec ndPulselength; // pulselength tp*kappa/w^2
    prec ndPulseOn; // pulse on time ton*kappa/w^2
    prec DutyCycle; // ton/tp

    string MoveName; // user file for laser movement
    TMoveArray ndMoveArray; // description of relative position
    bool bMoveLaser; // flag laser movement
    TPoint3D ndPosition; // (x,y,z)-Position of focus (scale w)
    TPoint3D ndInitialPos; // initial (x,y,z)-Position of focus
    // (scale w)
    TCPoint3D EField; // (complex) direction of electric field

    prec ndRelPower; // rel. intensity phi(t)
    prec ndAvePower; // 0.5*(phi(t-dt)+phi(t))

    prec ndPower; // power scale = 1/Nk
    prec ndRate; // vaporization rate scale (ideal Gauss)

    int iRays; // number of rays
    int iBadRays; // number of lost rays
    prec RadiusFactor;
    prec PowerFactor;
};

ostream &operator<<(ostream &, TLaser &);
istream &operator>>(istream &, TLaser &);

extern TLaser Laser, LaserOld;
#endif
