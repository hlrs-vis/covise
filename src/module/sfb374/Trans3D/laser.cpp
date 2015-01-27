/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          laser.cpp  -  laser settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "laser.h"
#include "material.h"
#include "solid.h"
#include "grid.h"
#include "simul.h"
#include "fortran.h"
#include "raytrace.h"
#include "main.h"

#ifdef GUI
#include "trans3d_guiview.h"
#endif

TLaser Laser, LaserOld;

// ***************************************************************************
// class for laser movement
// ***************************************************************************

// save class (temporary)

void TMoveVect::Save(int unit)
{
    unitwrite(unit, ndTime, "%le\t");
    unitwrite(unit, ptBegin.x, "%le\t");
    unitwrite(unit, ptBegin.y, "%le\t");
    unitwrite(unit, ptEnd.x, "%le\t");
    unitwrite(unit, ptEnd.y, "%le\t");
    unitwrite(unit, ndVelocity, "%le\t");
    unitwrite(unit, ndRadius, "%le\t");
    unitwrite(unit, Angle, "%le\n");
}

// read class (temporary)

void TMoveVect::Read(int unit)
{
    ndTime = readreal(unit);
    ptBegin.x = readreal(unit);
    ptBegin.y = readreal(unit);
    ptEnd.x = readreal(unit);
    ptEnd.y = readreal(unit);
    ndVelocity = readreal(unit);
    ndRadius = readreal(unit);
    Angle = readreal(unit);
}

// ***************************************************************************
// class for laser settings
// ***************************************************************************

// copy constructor

TLaser::TLaser(const TLaser &src)
{
    ProfileName = src.ProfileName;
    bChanged = src.bChanged;
    RefIntensity = src.RefIntensity;
    Fluence = src.Fluence;
    Radius = src.Radius;
    Divergence = src.Divergence;
    ndWavelength = src.ndWavelength;
    ShapeName = src.ShapeName;
    ndGeomShape = src.ndGeomShape;
    Mode = src.Mode;
    ndHeight = src.ndHeight;
    PulseName = src.PulseName;
    ndPulseShape = src.ndPulseShape;
    iPulseForm = src.iPulseForm;
    Pulselength = src.Pulselength;
    ndPulselength = src.ndPulselength;
    ndPulseOn = src.ndPulseOn;
    DutyCycle = src.DutyCycle;
    MoveName = src.MoveName;
    ndMoveArray = src.ndMoveArray;
    bMoveLaser = src.bMoveLaser;
    ndPosition = src.ndPosition;

    ndInitialPos = src.ndInitialPos;
    EField = src.EField;
    ndRelPower = src.ndRelPower;
    ndAvePower = src.ndAvePower;
    ndPower = src.ndPower;
    ndRate = src.ndRate;
    iRays = src.iRays;
    iBadRays = src.iBadRays;
}

// reset variables

void TLaser::Reset()
{
    ProfileName.empty();
    bChanged = false;
    RefIntensity = 1e9;
    Radius = 4e-5;
    Divergence = 0.05;
    ndWavelength = 1064e-9 / Radius;
    ShapeName.empty();
    Mode = TEM00;
    ndHeight = 1;
    PulseName.empty();
    iPulseForm = rectangular;
    Pulselength = 1e-3;
    DutyCycle = 1e-4;
    MoveName.empty();
    bMoveLaser = false;
    ndInitialPos = TPoint3D(1.5, 1.5, 0);
    EField = TCPoint3D(cmplx(1, 0), cmplx(0, 1), cmplx(0, 0));
    RadiusFactor = 1.0;
    PowerFactor = 1.0;
}

// ***************************************************************************
// save settings
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TLaser &l)
{
    TLaser src = l.GetDimensional();

    ps << endl;
    ps << "laser settings:" << endl;
    ps << "profile name:\t" << src.ProfileName << endl;
    ps << "average intensity [W/m2]:\t" << src.RefIntensity << endl;
    ps << "focus radius [m]:\t" << src.Radius << endl;
    ps << "beam divergence [rad]:\t" << src.Divergence << endl;
    ps << "wavelength [m]:\t" << src.ndWavelength << endl;
    ps << "beam geometry file:\t" << src.ShapeName << endl;
    ps << "beam geometry:\t" << src.Mode << endl;
    ps << src.ndGeomShape;
    ps << "rhomb height [m]:\t" << src.ndHeight << endl;
    ps << "pulse shape file:\t" << src.PulseName << endl;
    ps << "pulse shape:\t" << src.iPulseForm << endl;
    ps << src.ndPulseShape;
    ps << "pulse length [s]:\t" << src.Pulselength << endl;
    ps << "duty cycle:\t" << src.DutyCycle << endl;
    ps << "laser motion file:\t" << src.MoveName << endl;
    ps << "laser motion:\t" << src.bMoveLaser << endl;
    ps << src.ndMoveArray;
    ps << "initial position [m]:\t" << src.ndInitialPos << endl;
    ps << "electric field vector:\t" << src.EField << endl;
    return ps;
}

// ***************************************************************************
// read settings
//
// output:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

istream &operator>>(istream &ps, TLaser &src)
{
    bool b;
    ps >> checkstring("laser settings:", &b);
    if (!b)
    {
        ps.setstate(ios::failbit);
        return ps;
    }
    src.bChanged = false;
    ps >> tab;
    getline(ps, src.ProfileName);
    ps >> tab >> src.RefIntensity;
    ps >> tab >> src.Radius;
    ps >> tab >> src.Divergence;
    ps >> tab >> src.ndWavelength >> tab;
    getline(ps, src.ShapeName);
    ps >> tab >> src.Mode;
    ps >> endl >> src.ndGeomShape;
    ps >> tab >> src.ndHeight >> tab;
    getline(ps, src.PulseName);
    ps >> tab >> src.iPulseForm;
    ps >> endl >> src.ndPulseShape;
    ps >> tab >> src.Pulselength;
    ps >> tab >> src.DutyCycle >> tab;
    getline(ps, src.MoveName);
    ps >> tab >> src.bMoveLaser;
    ps >> endl >> src.ndMoveArray;
    ps >> tab >> src.ndInitialPos;
    ps >> tab >> src.EField >> endl;
    src.MakeNonDimensional();
    src.Update();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
// ***************************************************************************

TLaser TLaser::GetDimensional()
{
    TMoveVect tm;
    TLaser l = *this;
    int i, imax;

    l.ndWavelength *= RefLength;
    l.ndHeight *= RefLength;
    l.Pulselength = l.ndPulselength * RefTime;
    l.ndPulseOn *= RefTime;
    l.ndPosition *= RefLength;
    l.ndInitialPos *= RefLength;
    l.ndRelPower *= RefIntensity;
    l.ndAvePower *= RefIntensity;

    imax = ndMoveArray.size();
    for (i = 0; i < imax; i++)
    {
        tm = l.ndMoveArray[i];
        tm.ndTime *= RefTime;
        tm.ndRadius *= RefLength;
        tm.ndVelocity *= RefVelocity;
        tm.ptBegin *= RefLength;
        tm.ptEnd *= RefLength;
        l.ndMoveArray[i] = tm;
    }
    return l;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// output:    RefLength, RefTime, RefVelocity
//
// ***************************************************************************

void TLaser::MakeNonDimensional()
{
    TMoveVect tm;
    int i, imax;

    RefLength = Radius;
    RefVelocity = Material.RefDiffusivity / RefLength;
    RefTime = RefLength / RefVelocity;

    ndWavelength /= RefLength;
    ndHeight /= RefLength;
    ndPulselength = Pulselength / RefTime;
    ndPulseOn /= RefTime;
    ndPosition /= RefLength;
    ndInitialPos /= RefLength;
    ndRelPower /= RefIntensity;
    ndAvePower /= RefIntensity;

    imax = ndMoveArray.size();
    for (i = 0; i < imax; i++)
    {
        tm = ndMoveArray[i];
        tm.ndTime /= RefTime;
        tm.ndRadius /= RefLength;
        tm.ndVelocity /= RefVelocity;
        tm.ptBegin /= RefLength;
        tm.ptEnd /= RefLength;
        ndMoveArray[i] = tm;
    }
}

// ***************************************************************************
// update dependent variables
//
// input:   Radius, Fluence, ShapeName, Mode, ndHeight, PulseName, iPulseForm,
//          Pulselength, DutyCycle, MoveName, iMove, ndInitialPos, Material
// output:  RefIntensity, ndGeomShape, ndPulseShape, ndPulseOn, ndPulselength,
//          ndMoveArray, ndPosition, ndRelPower, RefLength, RefTime

//          RefVelocity, RefPower, RefRate, ndVapRate
//
// call after material update!
//
// ***************************************************************************

void TLaser::Update()
{
    prec newLength, newFluence, newDuty;

    RefLength = Laser.Radius;
    RefVelocity = Material.RefDiffusivity / RefLength;
    RefTime = RefLength / RefVelocity;

    prec dT = Material.VaporTemp - Material.RoomTemp;

    ReadPulse(&newLength, &newFluence, &newDuty);
    if (newDuty != 0)
        DutyCycle = newDuty;
    ndPulselength = Pulselength / RefTime;
    if (iPulseForm == cw) // cw
        ndPulseOn = 1;
    else
        ndPulseOn = ndPulselength * DutyCycle;
    Fluence = RefIntensity * Pulselength;

    ndPower = Laser.RefIntensity * RefLength / (Material.RefConductivity * Material.RefTemp);
    ndRate = PI_OVER_TWO * ndPower / (1 + Material.RefStefan);
    Material.ndVapRate = 0.851 * 1.e5 / (Material.RefDensity * sqrt(TWO_PI * Material.SpecGasConst * dT) * RefVelocity);

    ReadShape();
    ReadMove();

    GetPower(Simulation.ndPulseTime); // initial power
    GetPosition(Simulation.ndTime); // initial position
}

// ***************************************************************************
// setup geometry
//
// parameters:    pointer to radius and intensity variables (if desired)
// return values: scaling parameter from file for radius and intensity
// input:         ShapeName, Mode, ndLength
// output:        ndGeomShape, Mode
//
// ***************************************************************************

void TLaser::ReadShape(prec *pRad, prec *pInt)
{
    int i, j, imax, jmax;
    prec dx, dy, r, q;

    if (pRad != 0)
        *pRad = 0;
    if (pInt != 0)
        *pInt = 0;
    if (Mode != user_mode || ShapeName.empty())
        return;
    ifstream is(ShapeName.c_str());
    ndGeomShape.Delete();
    if (!is) // file not found
    {
        *perr << "cannot open " << ShapeName.c_str() << endl;
        Mode = TEM00;
        return;
    }
    is >> imax >> tab >> jmax >> endl; // number of values
    is >> dx >> tab >> dy >> endl; // distances between values
    is >> r >> endl; // beam radius suggestion
    if (!is) // wrong format
    {
        *perr << "wrong file format in " << ShapeName.c_str() << endl;
        Mode = TEM00;
        return;
    }
    ndGeomShape.ReSize(imax, jmax);
    ndGeomShape.SetDx(dx / RefLength); // normalize with respect to radius
    ndGeomShape.SetDy(dy / RefLength);
    i = 0;
    while (i < imax)
    {
        j = 0;
        while (j < jmax)
        {
            is >> q;
            ndGeomShape.Set(i, j, q);
            j++;
        }
        i++;
    }
    q = ndGeomShape.GetPrec(0., 0.); // central intensity
    if (q == 0)
        q = 1;
    i = 0;
    while (i < imax)
    {
        j = 0;
        while (j < jmax)
        {
            // normalize Icenter = 1
            ndGeomShape.Set(i, j, ndGeomShape.Get(i, j) / q);
            j++;
        }
        i++;
    }
    if (!is) // wrong format
    {
        Mode = TEM00;
        *perr << "wrong file format in " << ShapeName.c_str() << endl;
        ndGeomShape.Delete();
        return;
    }
    if (pRad != 0)
        *pRad = r;
    if (pInt != 0)
        *pInt = q;
}

// ***************************************************************************
// setup movement
//
// input:         MoveName, bMoveLaser, RefLength, RefTime, RefVelocity
// output:        ndMoveArray, bMoveLaser
//
// ***************************************************************************

void TLaser::ReadMove()
{
    TMoveVect tm;

    if (!bMoveLaser || MoveName.empty())
        return;
    if (unitopen(15, MoveName.c_str()) < 0) // file not found
        return;
    ndMoveArray.clear();
    while ((tm.ndTime = readreal(15)) != -1)
    {
        tm.ndTime /= RefTime;
        tm.ptBegin.x = readreal(15) / RefLength; // first position
        tm.ptBegin.y = readreal(15) / RefLength;
        tm.ptEnd.x = readreal(15) / RefLength; // second position
        tm.ptEnd.y = readreal(15) / RefLength;
        tm.ndVelocity = readreal(15) / RefVelocity;
        tm.ndRadius = readreal(15) / RefLength;
        if (tm.ndRadius != 0) // calc initial angle for trepanning if radius not zero
        {
            tm.iType = TMoveVect::trepanning;
            tm.Angle = atan2(tm.ptBegin.y - tm.ptEnd.y, tm.ptBegin.x - tm.ptEnd.x);
        }
        else
            // calc distance
            tm.ndRadius = (tm.ptBegin - tm.ptEnd).Abs();
        ndMoveArray.push_back(tm);
    }
    unitclose(15);
}

// ***************************************************************************
// setup pulse shape
//
// parameters:    pointer to Pulselength, Fluence and Dutycycle variables
// return values: scaling parameter from file for Pulselength, Fluence and
//                Dutycycle
// input:         PulseName, iPulseForm, ndTime,
// output:        ndPulseShape, Fluence
//
// ***************************************************************************

void TLaser::ReadPulse(prec *pLen, prec *pFlu, prec *pDuty)
{
    int i, imax;
    prec x, y, l, f, d;

    if (pLen != 0)
        *pLen = 0;
    if (pFlu != 0)
        *pFlu = 0;
    if (pDuty != 0)
        *pDuty = 0;
    if (iPulseForm != user_form || PulseName.empty())
        return;
    if (unitopen(15, PulseName.c_str()) < 0) // file not found
        return;
    ndPulseShape.Delete();
    l = readreal(15) / RefTime; // suggestion pulselength
    while ((x = readreal(15)) != -1)
    {
        x /= RefTime;
        d = x; // pulse on time
        y = readreal(15);
        ndPulseShape.Set(x, y);
    }
    unitclose(15);
    f = ndPulseShape.Integrate(0, d); // suggestion fluence
    d /= l; // suggestion duty-cycle

    imax = ndPulseShape.GetEntries();
    for (i = 0; i < imax; i++) // normalize
        ndPulseShape.SetY(i, ndPulseShape.GetY(i) * ndPulselength / f);
    if (pLen != 0)
        *pLen = l; // nd
    if (pFlu != 0)
        *pFlu = f * RefTime; // [W/m^2]
    if (pDuty != 0)
        *pDuty = d;
}

// ***************************************************************************
// check if time of entry is smaller than that of pentry
// ***************************************************************************

bool MvSmallerEqual(const TMoveVect &entry, void *pentry)
{
    return (entry.ndTime <= ((TMoveVect *)pentry)->ndTime);
}

// ***************************************************************************
// get relative laser position
//
// parameters:    time
// return values: relative coordinates of laser
// input:         ndMoveArray, bMoveLaser
//
// ***************************************************************************

TPoint TLaser::GetMovement(prec t)
{
    TMoveVect tm;
    prec phi;

    if (!bMoveLaser)
        return TPoint(0, 0);
    tm.ndTime = t;
    tm = ndMoveArray.LastCondition(MvSmallerEqual, tm);
    if (tm.ndVelocity == 0 || tm.ndRadius == 0) // no movement
        return TPoint(0, 0);
    phi = tm.ndVelocity / tm.ndRadius * (t - tm.ndTime);
    if (tm.iType == TMoveVect::linear) // linear
        return tm.ptBegin + (tm.ptEnd - tm.ptBegin) * phi;
    phi += tm.Angle;
    // trepanning
    return tm.ptEnd + tm.ndRadius * TPoint(cos(phi), sin(phi));
}

// ***************************************************************************
// get laser position
//
// parameters:    nd time
// return values: absoulute coordinates of laser ndPosition
// output:        ndPosition
// input:         ndMoveArray, iMove, ndInitialPos
//
// ***************************************************************************

TPoint3D TLaser::GetPosition(prec t)
{
    TPoint pt = GetMovement(t);
    ndPosition = ndInitialPos + TPoint3D(pt.x, pt.y, 0);
    //ndPosition = ndInitialPos+TPoint3D(0.2,0.1,0.)*t/Simulation.ndDeltatOn;
    return ndPosition;
}

// ***************************************************************************
// save laser settings (temporary)
// ***************************************************************************

void TLaser::Save(int unit)
{
    unitwriteln(unit, "\nLaser:"); // nur Text
    unitwrite(unit, "Profilname =\t");
    unitwriteln(unit, ProfileName.c_str());

    unitwrite(unit, RefIntensity, "Intensität =\t%le\n");
    unitwrite(unit, Divergence, "Divergenz =\t%le\n");
    unitwrite(unit, ndWavelength * RefLength, "Wellenlänge =\t%le\n");
    unitwrite(unit, Radius, "Strahlradius =\t%le\n");
    unitwrite(unit, ndInitialPos.z * RefLength, "Fokuslage =\t%le\n");

    unitwrite(unit, "Geometriedatei =\t");
    unitwriteln(unit, ShapeName.c_str());
    if (!ShapeName.empty())
        ndGeomShape.Save(unit);
    unitwrite(unit, Mode, "Mode =\t%le\n");
    unitwrite(unit, ndHeight * RefLength, "Pulshöhe =\t%le\n");

    unitwrite(unit, "Pulsdatei =\t");
    unitwriteln(unit, PulseName.c_str());
    if (!PulseName.empty())
        ndPulseShape.Save(unit);
    unitwrite(unit, iPulseForm, "Pulsverlauf =\t%i\n");
    unitwrite(unit, Pulselength, "Pulslänge =\t%le\n");
    unitwrite(unit, DutyCycle, "DutyCycle =\t%le\n");

    unitwrite(unit, "Bewegungsdatei =\t");
    unitwriteln(unit, MoveName.c_str());
    unitwrite(unit, bMoveLaser, "Laserbewegung =\t%i\n");
    ndMoveArray.Save(unit);

    unitwrite(unit, int(0), "Polarisation =\t%i\n");
    unitwrite(unit, EField.x.real(), "Feldvektor real =\t%le");
    unitwrite(unit, EField.y.real(), "\t%le");
    unitwrite(unit, EField.z.real(), "\t%le\n");
    unitwrite(unit, EField.x.imag(), "Feldvektor imaginär =\t%le");
    unitwrite(unit, EField.y.imag(), "\t%le");
    unitwrite(unit, EField.z.imag(), "\t%le\n");
}

// ***************************************************************************
// read laser settings (temporary)
// ***************************************************************************

void TLaser::Read(int unit, float)
{
    char buffer[100];

    unitreadln(unit, buffer); // /n
    unitreadln(unit, buffer); // Laser
    ProfileName = unitreadln(unit, buffer);
    RefIntensity = readreal(unit);
    Divergence = readreal(unit);
    ndWavelength = readreal(unit);
    Radius = readreal(unit);
    ndWavelength /= Radius;
    ndInitialPos.z = readreal(unit) / Radius;
    ShapeName = unitreadln(unit, buffer);
    if (!ShapeName.empty())
        ndGeomShape.Read(unit);
    Mode = readreal(unit);
    ndHeight = readreal(unit) / Radius;
    PulseName = unitreadln(unit, buffer);
    if (!PulseName.empty())
        ndPulseShape.Read(unit);
    iPulseForm = readint(unit);
    Pulselength = readreal(unit);
    DutyCycle = readreal(unit);
    MoveName = unitreadln(unit, buffer);
    bMoveLaser = readbool(unit);
    ndMoveArray.Read(unit);
    readint(unit); // iPolType
    prec val[6];
    int i;
    for (i = 0; i < 6; i++)
        val[i] = readreal(unit);
    cmplx x(val[0], val[3]);
    cmplx y(val[1], val[4]);
    cmplx z(val[2], val[5]);
    EField.x = x;
    EField.y = y;
    EField.z = z;

    Update();
}

// ***************************************************************************
// get temporal laser flux
//
// parameters:    time relative to pulse start
// return values: flux ndRelPower
// output:        ndRelPower
// input:         PulseShape, iPulseForm
//
// ***************************************************************************

prec TLaser::GetPower(prec t)
{
    if (iPulseForm == cw) // continuous wave
    {
        ndRelPower = 1;
        return ndRelPower;
    }
    if (t > ndPulseOn) // after pulse
        ndRelPower = 0;
    else if (iPulseForm == rectangular) // rectangular pulse
        ndRelPower = 1 / DutyCycle;

    else // user defined
        ndRelPower = ndPulseShape.Get(t);

    return ndRelPower * PowerFactor;
}

// ***************************************************************************
// get local laser flux and direction (spacial and temporal)
//
// parameters:    position and flux
// return values: flux and direction
// input:         ndGeomShape, Mode, ndPosition, ndAvePower, Divergence...
//
// ***************************************************************************

prec TLaser::GetFlux(const TPoint3D &pos, TPoint3D &vflux)
{
    TPoint diffvec, pt;
    prec zorigin, waist2, flux, x2, y2, r2, rcurve, rarg, exparg;

    vflux = 0;
    if (pos.z >= Grid.ndWpThickness) // beyond workpiece
        return 0;
    diffvec.x = pos.x - ndPosition.x;
    diffvec.y = pos.y - ndPosition.y;

    zorigin = pos.z + ndPosition.z; // orgin of geometrical beam (gaussian prop)
    if (Mode >= THround && Mode <= THrhomb) // top-hat (straight propagation)
        waist2 = sqr(RadiusFactor * (fabs(Divergence * zorigin) + 1));
    else
        // square of relative
        waist2 = sqr(RadiusFactor * Divergence * zorigin) + 1;
    // beam waist w/w0 i.e. area

    x2 = sqr(diffvec.x);
    y2 = sqr(diffvec.y);
    r2 = x2 + y2;
    switch ((int)Mode)
    {
    case THround: // round top-hat
        flux = 1 / 1.32 * (exp(-16 * sqr(r2 + 0.7)) + exp(-16 * sqr(r2 - 0.7)) + exp(-13.5 * sqr(r2)) / 1.1 + exp(-13 * sqr(r2 + 0.35)) + exp(-13 * sqr(r2 - 0.35))) / waist2;
        break;

    case THrhomb: // rhomb top-hat
        x2 = sqr(diffvec.x / ndHeight + diffvec.y); // rotate and stretch
        y2 = sqr(diffvec.x / ndHeight - diffvec.y);

    case THsquare: // square top-hat
        flux = ((exp(-16 * sqr(y2 + 0.7)) + exp(-16 * sqr(y2 - 0.7)) + exp(-13.5 * sqr(y2)) / 1.1
                 + exp(-13 * sqr(y2 + 0.35)) + exp(-13 * sqr(y2 - 0.35)))
                * (exp(-16 * sqr(x2 + 0.7)) + exp(-16 * sqr(x2 - 0.7)) + exp(-13.5 * sqr(x2)) / 1.1
                   + exp(-13 * sqr(x2 + 0.35)) + exp(-13 * sqr(x2 - 0.35)))) / (sqr(1.32) * waist2);
        break;

    case user_mode:
        pt = diffvec / sqrt(waist2); // stretch coordinates
        flux = ndGeomShape.GetPrec(pt.x, pt.y) / waist2;
        break;

    default: // TEM
        exparg = 2 * r2 / waist2;
        flux = (Mode + (1 - Mode) * exparg) * exp(-exparg) / waist2;
        break;
    }

    flux *= ndAvePower; // temporal part
    vflux.Set(0, 0, flux);
    if (flux < 1e-10) // too small
        return flux;

    rcurve = sqr(Divergence) * zorigin / waist2; // one over radius of curvature
    vflux.x = diffvec.x * rcurve; // direction of ray
    vflux.y = diffvec.y * rcurve;
    rarg = 1 - sqr(vflux.x) - sqr(vflux.y);
    if (rarg <= 0)
        return flux;
    vflux.z = flux / sqrt(rarg);
    vflux.x *= vflux.z;
    vflux.y *= vflux.z;
    vflux.z = flux;

    return flux;
}

// ***************************************************************************
// calculate laser absorption in solid
//
// output:        TSolid: ndHeat; TSolSurface: ndDirAbs, ndMultAbs,
//                ndIncident, Normal, ndArea, ndTotalIn, ndTotalDirAbs,
//                ndTotalMultAbs, ndTotalTrans
// input:         almost all TLaser variables
//
// ***************************************************************************

void TLaser::CalcAbsorbedHeat(void)
{
    RayTrace tr, trold;
    TPoint pos;
    int i, j, k, im1, ip1, jm1, jp1, joddeven,
        ipatch, jpatch, iu, jv, numrays;
    prec ratio, reffrac, influx, fparin, fperin, fparref, fperref;

    if (Grid.iGridMove == TGrid::small_change && (ndPosition - LaserOld.ndPosition).Abs() == 0 && Material.ndIndex.GetEntries() < 2) // almost same geometry and no
    // temperature dependent absorption
    {
        if (Laser.iPulseForm == cw) // cw Mode => same absorption
            return;
        if (LaserOld.ndAvePower < 1e-3 && Laser.ndAvePower < 1e-3)
            return; // negligable heat input => done
        else
        { // simply rescale
            // scaling factor
            ratio = Laser.ndAvePower / LaserOld.ndAvePower;
            SolSurf.ndTotalTrans *= ratio;
            SolSurf.ndTotalIn *= ratio;
            SolSurf.ndTotalDirAbs *= ratio;
            SolSurf.ndTotalMultAbs *= ratio;
            for (i = 1; i <= Grid.iVolNodes; i++)
                for (j = 1; j <= Grid.jVolNodes; j++)
                {
                    SolSurf(i, j).ndIncident *= ratio;
                    SolSurf(i, j).ndDirAbs *= ratio;
                    SolSurf(i, j).ndMultAbs *= ratio;
                    for (k = 1; k < Grid.kVolNodes; k++)
                        Solid(i, j, k).ndHeat *= ratio;
                }
            return;
        }
    }

#ifdef raytrace
    unitopen(9, "raytrace.dat", ios::out, true);
#endif

    Laser.iRays = 0; // initialize
    Laser.iBadRays = 0;
    SolSurf.ndTotalTrans = 0;
    SolSurf.ndTotalIn = 0;
    SolSurf.ndTotalDirAbs = 0;
    SolSurf.ndTotalMultAbs = 0;
    for (i = 1; i <= Grid.iVolNodes; i++)
    {
        im1 = i - 1;
        ip1 = i + 1;
        for (j = 1; j <= Grid.jVolNodes; j++)
        {
            jm1 = j - 1;
            jp1 = j + 1;
            SolSurf(i, j).ndArea = 1e-30;
            SolSurf(i, j).ndIncident = 0;
            SolSurf(i, j).ndDirAbs = 0;
            SolSurf(i, j).ndMultAbs = 0;
            SolSurf(i, j).ndParAbs = 0;
            SolSurf(i, j).ndPerAbs = 0;
            for (k = 1; k < Grid.kVolNodes; k++)
                Solid(i, j, k).ndHeat = 0;
            SolSurf.CalcInwardNormal(i, j, Solid); // calc inward surface normal
            tr.Normal = SolSurf(i, j).Normal;

            if (ndAvePower >= 1e-3) // laser on
            {
                tr.RayIn = Solid(i, j, 1).ndNode; // current surface position
                SolSurf.CalcSurfaceArea(i, j, Solid); // calculate area
                GetFlux(tr.RayIn, tr.InDir); // calc flux and direction
                influx = tr.InDir * tr.Normal; // incident flux on surface
                // count total power
                SolSurf.ndTotalIn += influx * SolSurf(i, j).ndCoarseArea;
                if (influx >= 1e-10) // considerable flux
                { // calc number of rays
                    numrays = int(influx * SolSurf(i, j).ndCoarseArea * TWO_OVER_PI * ndPulseOn / ndPulselength * Simulation.iAveRays);
                    ratio = sqrt(Solid(i, j, 1).xsai.Norm() / Solid(i, j, 1).xeta.Norm())
                            + 1e-10; // calc number of sub-patches
                    SolSurf(i, j).jNumRays = int(sqrt(numrays / ratio));
                    SolSurf(i, j).iNumRays = int(numrays * SolSurf(i, j).jNumRays);

                    if (Simulation.iAveRays <= 1) // only coarse calculation necessary
                    {
                        // drilled through
                        if (tr.RayIn.z >= Grid.ndWpThickness)
                            SolSurf.ndTotalTrans += influx * SolSurf(i, j).ndCoarseArea;
                        else
                        {
                            if (Simulation.iReflection == TSimulation::polarized)
                                tr.EField = EField; // copy local electric field vector
                            tr.InDir.Normalize();
                            // reflected fraction
                            reffrac = Material.GetReflection(tr.InDir, tr.OutDir, tr.Normal,
                                                             tr.EField, Solid(i, j, 1).ndTemp, &fparin,
                                                             &fperin, &fparref, &fperref);
                            // absorbed flux
                            tr.ndFlux = influx * (1 - reffrac);
                            SolSurf(i, j).ndIncident += influx;
                            SolSurf(i, j).ndDirAbs += tr.ndFlux;
                            if (Simulation.iReflection == TSimulation::polarized)
                            { // absorbed power of parallel and perpendicularly
                                // polarized component
                                SolSurf(i, j).ndParAbs += influx * (1 - fparref) * fparin;
                                SolSurf(i, j).ndPerAbs += influx * (1 - fperref) * fperin;
                            }
                            // power
                            influx *= SolSurf(i, j).ndCoarseArea;
                            // absorbed flux
                            tr.ndFlux = influx * (1 - reffrac);
                            SolSurf.ndTotalIn += influx;
                            SolSurf.ndTotalDirAbs += tr.ndFlux;
                            SolSurf(i, j).ndArea = SolSurf(i, j).ndCoarseArea;
                            tr.TraceSolid(i, j); // trace ray into material
                        }
                    }
                }
            }
        }
    }

    // no finer resolution
    if (Laser.ndAvePower < 1e-3 || Simulation.iAveRays <= 1)
        return;

    for (i = 1; i < Grid.iVolNodes; i++) // start with finer resolution
        for (j = 1; j < Grid.jVolNodes; j++)
        { // averaging ray numbers over cells
            SolSurf(i, j).iNumRays += SolSurf(i + 1, j).iNumRays + SolSurf(i, j + 1).iNumRays + SolSurf(i + 1, j + 1).iNumRays;
            SolSurf(i, j).iNumRays = (1 + (int(SolSurf(i, j).iNumRays) >> 3)) << 1;
            SolSurf(i, j).jNumRays += SolSurf(i + 1, j).jNumRays + SolSurf(i, j + 1).jNumRays + SolSurf(i + 1, j + 1).jNumRays;
            SolSurf(i, j).jNumRays = (1 + (int(SolSurf(i, j).jNumRays) >> 3)) << 1;
            tr.delta_uv16th = 0.25 / (SolSurf(i, j).iNumRays * SolSurf(i, j).jNumRays);
            joddeven = -1; // toggling flag
            for (jv = 1; jv <= SolSurf(i, j).jNumRays; jv++) // iterate surface patches
            {
                joddeven = -joddeven;
                // v position
                pos.y = double(2 * jv - 1) / SolSurf(i, j).jNumRays - 1.0;
                jpatch = int(j + 1 + pos.y); // node index j for absorption
                for (iu = 1; iu <= SolSurf(i, j).iNumRays; iu++)
                {
                    // u
                    pos.x = (double(2 * iu - 1) / SolSurf(i, j).iNumRays - 1.0) * joddeven;
                    ipatch = int(i + 1 + pos.x); // node index i for absorption
                    tr.CalcPatchVect(i, j, pos); // calc position and normal
                    tr.CalcPatchArea(i, j, pos); // calc sub-patch area tr.ndArea
                    SolSurf(ipatch, jpatch).ndArea += tr.ndArea;
                    // sum over area patches around node
                    GetFlux(tr.RayIn, tr.InDir); // calc flux and direction
                    influx = tr.InDir * tr.Normal * tr.ndArea;
                    // incident power on surface of patch
                    if (tr.RayIn.z >= Grid.ndWpThickness) // drilled through
                        SolSurf.ndTotalTrans += influx;
                    else
                    {
                        if (Simulation.iReflection == TSimulation::polarized)
                            tr.EField = EField; // copy local electric field vector
                        tr.InDir.Normalize();
                        // reflected fraction
                        reffrac = Material.GetReflection(tr.InDir, tr.OutDir, tr.Normal,
                                                         tr.EField, Solid(ipatch, jpatch, 1).ndTemp,
                                                         &fparin, &fperin, &fparref, &fperref);
                        tr.ndRefFlux = influx * reffrac; // reflected power
                        tr.ndFlux = influx * (1 - reffrac); // absorbed power
                        SolSurf(ipatch, jpatch).ndIncident += influx;
                        SolSurf(ipatch, jpatch).ndDirAbs += tr.ndFlux;
                        if (Simulation.iReflection == TSimulation::polarized)
                        { // absorbed power of parallel and perpendicularly
                            // polarized component
                            SolSurf(ipatch, jpatch).ndParAbs += influx * (1 - fparref) * fparin;
                            SolSurf(ipatch, jpatch).ndPerAbs += influx * (1 - fperref) * fperin;
                        }
                        SolSurf.ndTotalIn += influx;
                        SolSurf.ndTotalDirAbs += tr.ndFlux;
                        trold = tr;
                        tr.TraceSolid(ipatch, jpatch); // trace ray into material
                        if (Simulation.bMultReflec && tr.ndRefFlux > Simulation.ndMinPower && tr.RayIn.z > Solid.ndMinSurZ)
                            trold.TraceGas(ipatch, jpatch);
                    }
                }
            }
        }
    for (i = 1; i <= Grid.iVolNodes; i++) // recalc intensities
        for (j = 1; j <= Grid.jVolNodes; j++)
        {
            SolSurf(i, j).ndIncident /= SolSurf(i, j).ndArea;
            SolSurf(i, j).ndDirAbs /= SolSurf(i, j).ndArea;
            SolSurf(i, j).ndMultAbs /= SolSurf(i, j).ndArea;
            SolSurf(i, j).ndParAbs /= SolSurf(i, j).ndArea;
            SolSurf(i, j).ndPerAbs /= SolSurf(i, j).ndArea;
        }
    return;
}
