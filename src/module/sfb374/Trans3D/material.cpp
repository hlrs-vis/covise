/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          material.cpp  -  material settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "material.h"
#include "laser.h"
#include "solid.h"
#include "grid.h"
#include "simul.h"
#include "fortran.h"
#include "main.h"

TMaterial Material;

int idum, iy; // for random number generation
ivector iv(ntab + 1);

// ***************************************************************************
// class for material settings
// ***************************************************************************

// copy constructor

TMaterial::TMaterial(const TMaterial &src)
{
    ProfileName = src.ProfileName;
    bChanged = src.bChanged;
    RoomTemp = src.RoomTemp;
    MeltTemp = src.MeltTemp;
    VaporTemp = src.VaporTemp;
    CriticalTemp = src.CriticalTemp;
    ndCriticalTemp = src.ndCriticalTemp;
    RefTemp = src.RefTemp;
    SpecGasConst = src.SpecGasConst;
    DensityName = src.DensityName;
    ndDensity = src.ndDensity;
    RefDensity = src.RefDensity;
    ConductName = src.ConductName;
    ndConductivity = src.ndConductivity;
    RefConductivity = src.RefConductivity;
    SpecHeatName = src.SpecHeatName;
    ndSpecHeat = src.ndSpecHeat;
    RefSpecHeat = src.RefSpecHeat;
    LatentMelt = src.LatentMelt;
    LatentVapor = src.LatentVapor;
    IndexName = src.IndexName;
    ndIndex = src.ndIndex;
    RefIndex = src.RefIndex;
    AbsCoeff = src.AbsCoeff;
    ndDiffusivity = src.ndDiffusivity;
    RefDiffusivity = src.RefDiffusivity;
    Kelvin = src.Kelvin;
    ndStefan = src.ndStefan;
    RefStefan = src.RefStefan;
    ndLatent = src.ndLatent;
    ndVapRate = src.ndVapRate;
}

// reset variables (Aluminum values)

void TMaterial::Reset()
{
    ProfileName.empty();
    bChanged = false;
    RoomTemp = 300;
    MeltTemp = 1000;
    VaporTemp = 2793;
    CriticalTemp = 8500;
    SpecGasConst = 308.3;
    RefDensity = 2385;
    DensityName.empty();
    ndDensity.Delete();
    RefConductivity = 94;
    ConductName.empty();
    ndConductivity.Delete();
    RefSpecHeat = 1080;
    SpecHeatName.empty();
    ndSpecHeat.Delete();
    LatentMelt = 3.9e5;
    LatentVapor = 1.08e7;
    IndexName.empty();
    ndIndex.Delete();
    RefIndex = cmplx(1.35, 9.58);
    AbsCoeff = 0.9;
    Update();
}

// ***************************************************************************
// save settings
//
// input:   RefLength, RefTime, RefPower
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TMaterial &mat)
{
    TMaterial src = mat.GetDimensional();

    ps << "material settings:" << endl;
    ps << endl;
    ps << "profile name:\t" << src.ProfileName << endl;
    ps << "room temperature [K]:\t" << src.RoomTemp << endl;
    ps << "boiling temperature [K]:\t" << src.VaporTemp << endl;
    ps << "critical temperature [K]:\t" << src.CriticalTemp << endl;
    ps << "specific gas constant [J/kg/K]:\t" << src.SpecGasConst << endl;
    ps << "density file:\t" << src.DensityName << endl;
    ps << "temperature [K]\tdensity [kg/m3]" << endl;
    ps << src.ndDensity;
    ps << "conductivity file:\t" << src.ConductName << endl;
    ps << "temperature [K]\tconductivity [W/m/K]" << endl;
    ps << src.ndConductivity;
    ps << "specific heat capacity file:\t" << src.SpecHeatName << endl;
    ps << "temperature [K]\tspecific heat capacity [J/kg/K]" << endl;
    ps << src.ndSpecHeat;
    ps << "latent heat of melting [J/kg]:\t" << src.LatentMelt << endl;
    ps << "latent heat of vaporization [J/kg]:\t" << src.LatentVapor << endl;
    ps << "refraction index file:\t" << src.IndexName << endl;
    ps << "temperature [K]\trefraction index" << endl;
    ps << src.ndIndex;
    ps << "absorption coefficient:\t" << src.AbsCoeff << endl;
    return ps << endl;
}

// ***************************************************************************
// read settings
//
// output:    RefDensity, RefConductivity, RefSpecHeat, RefIndex
//
// ***************************************************************************

istream &operator>>(istream &ps, TMaterial &src)
{
    src.bChanged = true;
    if (!CheckHeader(ps, "material settings:"))
        return ps;
    src.bChanged = false;
    ps >> tab;
    getline(ps, src.ProfileName);
    ps >> tab >> src.RoomTemp;
    ps >> tab >> src.VaporTemp;
    ps >> tab >> src.CriticalTemp;
    ps >> tab >> src.SpecGasConst >> tab;
    getline(ps, src.DensityName);
    ps >> endl >> src.ndDensity >> tab;
    getline(ps, src.ConductName);
    ps >> endl >> src.ndConductivity >> tab;
    getline(ps, src.SpecHeatName);
    ps >> endl >> src.ndSpecHeat;
    ps >> tab >> src.LatentMelt;
    ps >> tab >> src.LatentVapor >> tab;
    getline(ps, src.IndexName);
    ps >> endl >> src.ndIndex;
    ps >> tab >> src.AbsCoeff >> endl >> endl;
    src.MakeNonDimensional();
    src.Update();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
//
// input:   RefDensity, RefConductivity, RefSpecHeat, Kelvin
//
// ***************************************************************************

TMaterial TMaterial::GetDimensional()
{
    TMaterial mat = *this;
    int i, imax;

    mat.ndDensity *= TPoint(1, RefDensity);
    mat.ndConductivity *= TPoint(1, RefConductivity);
    mat.ndSpecHeat *= TPoint(1, RefSpecHeat);

    imax = ndIndex.GetEntries();
    for (i = 0; i < imax; i++)
        mat.ndIndex.SetX(i, GetKelvin(ndIndex.GetX(i)));
    return mat;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// output:  RefDensity, RefConductivity, RefSpecHeat, RefTemp, RefIndex
// input:   VaporTemp, ndConductivity
//
// ***************************************************************************

void TMaterial::MakeNonDimensional()
{
    int i, imax;

    RefDensity = ndDensity.Get(VaporTemp);
    RefConductivity = ndConductivity.Get(VaporTemp);
    RefSpecHeat = ndSpecHeat.Get(VaporTemp);
    RefIndex = ndIndex.Get(VaporTemp);

    ndDensity /= TPoint(1, RefDensity);
    ndConductivity /= TPoint(1, RefConductivity);
    ndSpecHeat /= TPoint(1, RefSpecHeat);

    CalcRefTemp();
    imax = ndIndex.GetEntries();
    for (i = 0; i < imax; i++)
        ndIndex.SetX(i, GetKirchhoff(ndIndex.GetX(i)));
}

// ***************************************************************************
// update dependent variables
//
// input:   RoomTemp, VaporTemp, CriticalTemp; Density, Conductivity,
//          SpecificHeat (all: values, reference value or filename),
//          LatentVapor
// output:  Kelvin, ndCriticalTemp, ndDiffusivity, RefStefan, RefTemp,
//          ndTRoom, ndLatent, ndStefan
//
// ***************************************************************************

void TMaterial::Update()
{
    ReadConductivity();
    CalcRefTemp();
    ReadDensity();
    ReadSpecHeat();
    ReadIndex();

    CalcKelvin();
    CalcDiffusivity();
    CalcStefan();
    ndCriticalTemp = GetKirchhoff(CriticalTemp);

    RefStefan = LatentVapor / (RefSpecHeat * RefTemp);
    ndLatent = LatentVapor / (SpecGasConst * VaporTemp);
}

// ***************************************************************************
// read thermal conductivity for valid filename and scale
//
// input:   VaporTemp
//
// ***************************************************************************

void TMaterial::ReadConductivity()
{
    prec x, y;

    if (!ConductName.empty())
        if (unitopen(15, ConductName.c_str()) >= 0)
        {
            ndConductivity.Delete();
            while ((x = readreal(15)) != -1)
            {
                y = readreal(15);
                ndConductivity.Set(x, y);
            }
            unitclose(15);

            RefConductivity = ndConductivity.Get(VaporTemp);
            for (int i = 0; i < ndConductivity.GetEntries(); i++)
                ndConductivity.SetY(i, ndConductivity.GetY(i) / RefConductivity);
        }
    if (ndConductivity.GetEntries() == 0)
        ndConductivity.Set(1, 1);
}

// ***************************************************************************
// read density for valid filename and scale
//
// input:   VaporTemp
//
// ***************************************************************************

void TMaterial::ReadDensity()
{
    prec x, y;

    if (!DensityName.empty())
        if (unitopen(15, DensityName.c_str()) >= 0)
        {
            ndDensity.Delete();
            while ((x = readreal(15)) != -1)
            {
                y = readreal(15);
                ndDensity.Set(x, y);
            }
            unitclose(15);

            RefDensity = ndDensity.Get(VaporTemp);
            for (int i = 0; i < ndDensity.GetEntries(); i++)
                ndDensity.SetY(i, ndDensity.GetY(i) / RefDensity);
        }
    if (ndDensity.GetEntries() == 0)
        ndDensity.Set(1, 1);
}

// ***************************************************************************
// read specific heat for valid filename and scale
//
// input:   VaporTemp
//
// ***************************************************************************

void TMaterial::ReadSpecHeat()
{
    prec x, y;

    if (!SpecHeatName.empty())
        if (unitopen(15, SpecHeatName.c_str()) >= 0)
        {
            ndSpecHeat.Delete();
            while ((x = readreal(15)) != -1)
            {
                y = readreal(15);
                ndSpecHeat.Set(x, y);
            }
            unitclose(15);

            RefSpecHeat = ndSpecHeat.Get(VaporTemp);
            for (int i = 0; i < ndSpecHeat.GetEntries(); i++)
                ndSpecHeat.SetY(i, ndSpecHeat.GetY(i) / RefSpecHeat);
        }
    if (ndSpecHeat.GetEntries() == 0)
        ndSpecHeat.Set(VaporTemp, 1);
}

// ***************************************************************************
// read index of refraction for valid filename
//
// input:   ndConductivity, RefTemp, VaporTemp
//
// ***************************************************************************

void TMaterial::ReadIndex()
{
    prec x, y, z;

    if (!IndexName.empty())
        if (unitopen(15, IndexName.c_str()) >= 0)
        {
            ndIndex.Delete();
            while ((x = readreal(15)) != -1)
            {
                x = GetKirchhoff(x);
                y = readreal(15);
                z = readreal(15);
                ndIndex.Set(x, cmplx(y, z));
            }
            unitclose(15);

            RefIndex = ndIndex.Get(1);
        }
    if (ndIndex.GetEntries() == 0)
        ndIndex.Set(1, RefIndex);
}

// ***************************************************************************
// calculate nondimensional temperature
//
// input: ndConductivity, RefTemp
//
// ***************************************************************************

prec TMaterial::GetKirchhoff(prec t)
{
    if (ndConductivity.GetEntries() < 2)
        return (t - RoomTemp) / RefTemp;
    return ndConductivity.Integrate(RoomTemp, t) / RefTemp;
}

// ***************************************************************************
// calculate reference temperature
//
// output:  RefTemp
// input:   ndConductivity
//
// ***************************************************************************

void TMaterial::CalcRefTemp()
{
    if (ndConductivity.GetEntries() < 2)
        RefTemp = VaporTemp - RoomTemp;
    else
        RefTemp = ndConductivity.Integrate(RoomTemp, VaporTemp);
}

// ***************************************************************************
// calculate temperature interpolation
//
// output:  Kelvin
// input:   ndConductivity, RefTemp, VaporTemp
//
// ***************************************************************************

void TMaterial::CalcKelvin()
{
    int i, imax;
    prec t;

    Kelvin.Delete();
    if ((imax = ndConductivity.GetEntries()) > 0)
    {
        i = 0;
        if (ndConductivity.GetX(0) > RoomTemp)
            Kelvin.Set(0, RoomTemp);
        while (i < imax)
        {
            t = ndConductivity.GetX(i);
            Kelvin.Set(GetKirchhoff(t), t);
            i++;
        }
    }
    if (Kelvin.GetEntries() == 0)
        Kelvin.Set(1, VaporTemp);
}

// ***************************************************************************
// convert Kirchhoff temperature into Kelvin
//
// input: Conductivity, RefTemp, Kelvin
//
// ***************************************************************************

prec TMaterial::GetKelvin(prec theta)
{
    int imax;
    prec thetamax, tmax;

    if ((imax = ndConductivity.GetEntries()) < 2)
        return RoomTemp + theta * RefTemp;

    thetamax = Kelvin.GetX(imax - 1);
    tmax = Kelvin.GetY(imax - 1);
    if (theta <= thetamax)
        return Kelvin.Get(theta);

    return tmax + (theta - thetamax) * RefTemp / ndConductivity.Get(tmax);
}

// ***************************************************************************
// calculate diffusivity interpolation
//
// output:  ndDiffusivity, RefDiffusivity
// input:   ndConductivity, ndDensity, ndSpecHeat, RefTemp,
//          RefConductivity, RefDensity, RefSpecHeat
//
// ***************************************************************************

void TMaterial::CalcDiffusivity()
{
    int i, j, k, imax, jmax, kmax;
    prec t, p;

    imax = ndConductivity.GetEntries();
    jmax = ndSpecHeat.GetEntries();
    kmax = ndDensity.GetEntries();

    ndDiffusivity.Delete();
    i = 0;
    j = 0;
    k = 0;
    while (i < imax || j < jmax || k < kmax)
    {
        t = 1e10;
        if (i < imax)
            t = min(ndConductivity.GetX(i), t);
        if (j < jmax)
            t = min(ndSpecHeat.GetX(j), t);
        if (k < kmax)
            t = min(ndDensity.GetX(k), t);
        p = ndConductivity.Get(t) / (ndSpecHeat.Get(t) * ndDensity.Get(t));
        ndDiffusivity.Set(GetKirchhoff(t), p);
        while (ndConductivity.GetX(i) == t && i < imax)
            i++;
        while (ndSpecHeat.GetX(j) == t && j < jmax)
            j++;
        while (ndDensity.GetX(k) == t && k < kmax)
            k++;
    }
    RefDiffusivity = RefConductivity / (RefSpecHeat * RefDensity);
    if (ndDiffusivity.GetEntries() == 0)
        ndDiffusivity.Set(1, RefDiffusivity);
}

// ***************************************************************************
// calculate Stefan number interpolation
//
// output:  ndStefan, RefStefan
// input:   ndConductivity, ndDensity, RefSpecHeat, RefTemp, LatentVapor
//
// ***************************************************************************

void TMaterial::CalcStefan()
{
    int i, imax;
    prec t, p;

    RefStefan = LatentVapor / (RefSpecHeat * RefTemp);
    ndStefan.Delete();
    if ((imax = ndDensity.GetEntries()) > 0)
    {
        i = 0;
        while (i < imax)
        {
            t = ndDensity.GetX(i);
            p = ndDensity.GetY(i) * RefStefan;
            ndStefan.Set(GetKirchhoff(t), p);
            while (ndDensity.GetX(i) == t && i < imax)
                i++;
        }
    }
    if (ndStefan.GetEntries() == 0)
        ndStefan.Set(1, RefStefan);
}

// ***************************************************************************
// save material settings (temporary use)
// ***************************************************************************

void TMaterial::Save(int unit)
{
    int i;

    unitwriteln(unit, "\nMaterial:");
    unitwrite(unit, "Profilname =\t");
    unitwriteln(unit, ProfileName.c_str());
    unitwrite(unit, RoomTemp, "Raumtemperatur =\t%le\n");
    unitwrite(unit, MeltTemp, "Schmelztemperatur =\t%le\n");
    unitwrite(unit, VaporTemp, "Verdampfungstemperatur =\t%le\n");
    unitwrite(unit, CriticalTemp, "kritischeTemperatur =\t%le\n");
    unitwrite(unit, LatentMelt, "Schmelzenthalpie =\t%le\n");
    unitwrite(unit, LatentVapor, "Verdampfungsenthalpie =\t%le\n");
    unitwrite(unit, SpecGasConst, "spez. Gaskonstante =\t%le\n");
    unitwrite(unit, AbsCoeff, "Absorptionskoeffizient =\t%le\n");
    unitwrite(unit, "Dichtedatei =\t");
    unitwriteln(unit, DensityName.c_str());
    unitwrite(unit, "Waermedatei =\t");
    unitwriteln(unit, SpecHeatName.c_str());
    unitwrite(unit, "Leitfaehigkeitsdatei =\t");
    unitwriteln(unit, ConductName.c_str());
    unitwrite(unit, "Brechungsindexdatei =\t");
    unitwriteln(unit, IndexName.c_str());

    // temporary: save dimensional

    for (i = 0; i < ndDensity.GetEntries(); i++)
        ndDensity.SetY(i, ndDensity.GetY(i) * RefDensity);
    ndDensity.Save(unit);
    for (i = 0; i < ndDensity.GetEntries(); i++)
        ndDensity.SetY(i, ndDensity.GetY(i) / RefDensity);
    for (i = 0; i < ndConductivity.GetEntries(); i++)
        ndConductivity.SetY(i, ndConductivity.GetY(i) * RefConductivity);
    ndConductivity.Save(unit);
    for (i = 0; i < ndConductivity.GetEntries(); i++)
        ndConductivity.SetY(i, ndConductivity.GetY(i) / RefConductivity);
    for (i = 0; i < ndSpecHeat.GetEntries(); i++)
        ndSpecHeat.SetY(i, ndSpecHeat.GetY(i) * RefSpecHeat);
    ndSpecHeat.Save(unit);
    for (i = 0; i < ndSpecHeat.GetEntries(); i++)
        ndSpecHeat.SetY(i, ndSpecHeat.GetY(i) / RefSpecHeat);
    rinterpol ndIndexReal, ndIndexImag;
    int imax = ndIndex.GetEntries();
    cmplx c;
    prec p;
    for (i = 0; i < imax; i++)
    {
        p = GetKelvin(ndIndex.GetX(i));
        c = ndIndex.GetY(i);
        ndIndexReal.Set(p, c.real());
        ndIndexImag.Set(p, c.imag());
    }
    ndIndexReal.Save(unit);
    ndIndexImag.Save(unit);
}

// ***************************************************************************
// read material settings (temporary)
// ***************************************************************************

void TMaterial::Read(int unit, float)
{
    int i;
    char buffer[100];

    unitreadln(unit, buffer);
    unitreadln(unit, buffer);
    ProfileName = unitreadln(unit, buffer);
    RoomTemp = readreal(unit);
    MeltTemp = readreal(unit);
    VaporTemp = readreal(unit);
    CriticalTemp = readreal(unit);
    LatentMelt = readreal(unit);
    LatentVapor = readreal(unit);
    SpecGasConst = readreal(unit);
    AbsCoeff = readreal(unit);
    DensityName = unitreadln(unit, buffer);
    SpecHeatName = unitreadln(unit, buffer);
    ConductName = unitreadln(unit, buffer);
    IndexName = unitreadln(unit, buffer);

    // temporary: dimensional input

    ndDensity.Read(unit);
    RefDensity = ndDensity.Get(VaporTemp);
    for (i = 0; i < ndDensity.GetEntries(); i++)
        ndDensity.SetY(i, ndDensity.GetY(i) / RefDensity);
    ndConductivity.Read(unit);
    RefConductivity = ndConductivity.Get(VaporTemp);
    for (i = 0; i < ndConductivity.GetEntries(); i++)
        ndConductivity.SetY(i, ndConductivity.GetY(i) / RefConductivity);
    ndSpecHeat.Read(unit);
    RefSpecHeat = ndSpecHeat.Get(VaporTemp);
    for (i = 0; i < ndSpecHeat.GetEntries(); i++)
        ndSpecHeat.SetY(i, ndSpecHeat.GetY(i) / RefSpecHeat);
    CalcRefTemp();
    rinterpol ndIndexReal, ndIndexImag;
    ndIndexReal.Read(unit);
    ndIndexImag.Read(unit);
    int imax = ndIndexReal.GetEntries();
    cmplx c;
    prec p;
    ndIndex.Delete();
    for (i = 0; i < imax; i++)
    {
        p = GetKirchhoff(ndIndexReal.GetX(i));
        c = cmplx(ndIndexReal.GetY(i), ndIndexImag.GetY(i));
        ndIndex.Set(p, c);
    }
    RefIndex = ndIndex.Get(1);
    Update();
}

// ***************************************************************************
// calculate random number (temporary)
// ***************************************************************************

float ran1(void)
{
    double ran1;
    const int ia = 16807, im = 2147483647, iq = 127773, ir = 2836, ndiv = 1 + (im - 1) / ntab;
    const prec am = 1.0 / im, eps = 1.2e-7, rnmx = 1.0 - eps;
    int j, k;

    if (idum < 0 || iy == 0)
    {
        idum = max(-idum, 1);
        for (j = ntab + 8; j >= 1; j--)
        {
            k = idum / iq;
            idum = ia * (idum - k * iq) - ir * k;
            if (idum < 0)
                idum += im;
            if (j <= ntab)
                iv(j) = idum;
        }
        iy = iv(1);
    }

    k = idum / iq;
    idum = ia * (idum - k * iq) - ir * k;
    if (idum < 0)
        idum += im;
    j = 1 + iy / ndiv;
    iy = iv(j);
    iv(j) = idum;
    ran1 = min(am * iy, rnmx);
    return ran1;
}

// ***************************************************************************
// get absorption index and optical penetration depth
//
// parameters:      temperature, complex index
// return values:   complex index, optical penetration depth
// input:           ndIndex, Laser.ndWavelength
//
// ***************************************************************************

prec TMaterial::GetAbsIndex(prec theta, cmplx &index)
{
    index = ndIndex.Get(theta);
    return 2 * TWO_PI * index.imag() / Laser.ndWavelength;
}

// ***************************************************************************
// calculate reflected part of incident intensity
//
// parameters:      incident and outgoing direction, surface normal,
//                  temperature, electric field vector, [parallel and
//                  perpendicular intensity component pointer]
// return values:   outgoing direction, [parallel and perpendicular intensity
//                  component pointer], reflection coefficient
// input:           ndIndexReal, ndIndexImag
//
// ***************************************************************************

prec TMaterial::GetReflection(TPoint3D &dirin, TPoint3D &dirout,
                              TPoint3D &surfnorm, TCPoint3D &EField,
                              prec theta, prec *pparin, prec *pperin,
                              prec *pparref, prec *pperref)
{
    cmplx EFieldInPar, EFieldInPer, EFieldOutPar, EFieldOutPer;
    prec incos, insin2, insintan, tmp1, tmp2, tmp3, tmp4, n2, k2,
        p, q, p2, q2;
    cmplx index, ctmp1, ctmp2;
    TPoint3D vecper, invecpar, outvecpar;

    if (pparin != 0)
        *pparin = 0;
    if (pperin != 0)
        *pperin = 0;
    if (pparref != 0)
        *pparref = 0;
    if (pperref != 0)
        *pperref = 0;
    GetAbsIndex(theta, index);
    incos = dirin * surfnorm; // cosine of incident angle
    if (!Simulation.bSpecReflec) // diffuse case
    {
        prec sin2 = ran1(); // random sin^2 for polar angle
        prec sinth = sqrt(sin2); // sine of polar angle
        prec costh = sqrt(1.0 - sin2); // cosine of polar angle
        prec psi = TWO_PI * ran1(); // random azimuthal angle
        prec sc = sinth * cos(psi);
        prec ss = sinth * sin(psi);
        prec cc = sqrt(sqr(surfnorm.y) + sqr(surfnorm.z));

        dirout.x = -costh * surfnorm.x - ss * cc;
        dirout.y = -costh * surfnorm.y + (sc * surfnorm.z + ss * surfnorm.x * surfnorm.y) / cc;
        dirout.z = -costh * surfnorm.z - (sc * surfnorm.y - ss * surfnorm.x * surfnorm.z) / cc;
    }
    else // specular reflection
        dirout = dirin - 2 * surfnorm * incos; // outgoing direction

    // constant absorption
    if (Simulation.iReflection == TSimulation::constant)
        return 1. - Material.AbsCoeff;

    // Fresnel absorption

    if (incos >= 1) // normal incident direction
    {
        incos = 1;
        insintan = 0;
        insin2 = 0;
    }
    else if (incos <= 0) // parallel to surface: all reflected
        return 1;
    else
    {
        insin2 = 1 - sqr(incos); // sin(theta)^2
        insintan = insin2 / incos; // sin(theta)*cos(theta);
    }

    n2 = sqr(index.real());
    k2 = sqr(index.imag());
    tmp1 = n2 - k2 - insin2;
    tmp2 = sqr(tmp1) + 4 * n2 * k2;
    p2 = 0.5 * (sqrt(tmp2) + tmp1);
    q2 = 0.5 * (sqrt(tmp2) - tmp1);
    p = sqrt(p2);
    q = sqrt(q2);

    // unpolarized
    if (Simulation.iReflection == TSimulation::unpolarized)
    {
        tmp1 = (sqr(incos - p) + q2) / (sqr(incos + p) + q2); // perp reflection
        // par refl.
        tmp2 = tmp1 * (sqr(p - insintan) + q2) / (sqr(p + insintan) + q2);
        return 0.5 * (tmp1 + tmp2); // return average
    }
    // polarized

    if (insin2 < 1.e-8) // small angle of incidence
    {
        tmp1 = sqrt(sqr(dirin.y) + sqr(dirin.z)); // take e1 x dirin as plane
        vecper.Set(0, -dirin.z / tmp1, dirin.y / tmp1);
    }
    else // vector perpendicular to plane of incidence
        vecper = -CrossProduct(dirin, surfnorm) / sqrt(insin2);
    invecpar = CrossProduct(vecper, dirin); // vector in plane of incidence

    EFieldInPar = EField * invecpar;
    EFieldInPer = EField * vecper;
    outvecpar = (dirout * invecpar) * dirin - (dirin * dirout) * invecpar;

    ctmp1 = cmplx(incos - p, q);
    ctmp2 = cmplx(incos + p, -q);
    EFieldOutPer = EFieldInPer * (ctmp1 / ctmp2);

    ctmp1 = cmplx(p - insintan, q);
    ctmp2 = cmplx(p + insintan, q);
    EFieldOutPar = EFieldInPar * (ctmp1 / ctmp2);

    EField = EFieldOutPar * outvecpar + EFieldOutPer * vecper;

    tmp1 = norm(EFieldOutPar);
    tmp2 = norm(EFieldOutPer);
    tmp3 = norm(EFieldInPar);
    tmp4 = norm(EFieldInPer);
    if (pparref != 0 && tmp3 != 0) // reflected parallel fraction
        *pparref = tmp1 / tmp3;
    if (pperref != 0 && tmp4 != 0) // reflected perpendicular fraction
        *pperref = tmp2 / tmp4;
    tmp3 += tmp4;
    tmp1 /= tmp3;
    tmp2 /= tmp3;
    if (pparin != 0) // fraction of parallel incident intensity
        *pparin = tmp1;
    if (pperin != 0) // fraction of perpendicular incident intensity
        *pperin = tmp2;
    return min(tmp1 + tmp2, 1.);
}

// ***************************************************************************
// get evaporation rate
//
// parameters:      temperature
// return values:   velocity
// input:           ndVapRate, ndLatent, ndTRoom
//
// ***************************************************************************

prec TMaterial::GetEvapRate(prec theta)
{
    prec temp = GetKelvin(theta) / VaporTemp;
    return ndVapRate / sqrt(temp) * exp(ndLatent * (1 - 1. / temp));
}

// ***************************************************************************
// get evaporation velocity
//
// parameters:      temperature
// return values:   velocity
// input:           ndVapRate, ndLatent, ndStefan, RefStefan, Kelvin, ...
//
// ***************************************************************************

prec TMaterial::GetEvapVelocity(prec theta)
{
    prec rhoratio = ndStefan.Get(theta) / RefStefan;
    return GetEvapRate(theta) / rhoratio;
}

// ***************************************************************************
// surface temperature for given velocity
//
// output:    surface temperature theta for ablation velocity vn
// input:     velocity vn, temperature estimate theta0, flag if
//            exact solution is desired
//
// ***************************************************************************

prec TMaterial::GetSurfTemp(prec vn, prec theta0, bool bExact)
{
    prec rhoratio, theta, temp;

    temp = GetKelvin(theta0) / VaporTemp;
    rhoratio = ndStefan.Get(theta0) / RefStefan;
    temp = VaporTemp / (1 - log(rhoratio * vn * sqrt(temp) / ndVapRate) / ndLatent);
    theta = GetKirchhoff(temp);
    if (!bExact)
        return theta; // only estimation

    prec f0, f1, df;

    theta0 = theta;
    f0 = GetEvapVelocity(theta0) - vn;
    do
    {
        f1 = GetEvapVelocity(theta0 + Simulation.ndTempTol) - vn;
        df = Simulation.ndTempTol / (f1 / f0 - 1);
        do // underrelaxation
        {
            theta = theta0 - df;
            f1 = GetEvapVelocity(theta) - vn;
            if (fabs(df) < Simulation.ndTempTol)
            {
                ErrorFunction("ERROR: no convergence for surface temperature");
                return -1;
            }
            df /= 2;
        } while (fabs(f1) > fabs(f0));
        f0 = f1;
        theta0 = theta;
    } while (fabs(f0) > vn * 1e-3);

    return theta;
}
