/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          material.h  -  material settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __MATERIAL_H_

#define __MATERIAL_H_

#ifndef __sgi
#include <cstring>
#endif

#include "classext.h"

const int ntab = 32; // variables for random number
extern int idum, iy; // (obsolete)
extern ivector iv;

// ***************************************************************************
// material settings
// ***************************************************************************

class TMaterial
{
public:
    TMaterial()
    {
        Reset();
    }
    TMaterial(const TMaterial &);

    void Reset(); // reset values
    void Update(); // update values

    void Save(int); // save settings
    void Read(int, float vers = 1.0); // read settings
    void ReadDensity(); // read density
    void ReadSpecHeat(); // read specific heat
    void ReadConductivity(); // read thermal conductivity
    void ReadIndex(); // read index of refraction
    TMaterial GetDimensional(); // all values in dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form

    prec GetKirchhoff(prec); // get Kirchhoff
    prec GetKelvin(prec); // get Kelvin
    prec GetAbsIndex(prec, cmplx &); // get index of refraction
    prec GetReflection(TPoint3D &, TPoint3D &, TPoint3D &, TCPoint3D &,
                       prec, prec *pparin = 0, prec *pperin = 0,
                       prec *pparref = 0, prec *pperref = 0);
    // calculate reflected part
    prec GetEvapRate(prec); // calculate evaporation rate
    prec GetEvapVelocity(prec); // calc evaporation velocity
    prec GetSurfTemp(prec, prec,
                     bool bExact = false); // calc surface temperature

    void CalcRefTemp(); // calc ndTemp
    void CalcKelvin(); // calc Kelvin interpolation
    void CalcDiffusivity(); // calc diffusivity interpolation
    void CalcStefan(); // calc diffusivity interpolation

    string ProfileName; // name of used profile
    bool bChanged; // change flag

    prec RoomTemp; // room temperature
    prec MeltTemp; // melting temperature
    prec VaporTemp; // vaporization temperature
    prec CriticalTemp; // critical temperature
    prec ndCriticalTemp; // critical temperature
    prec RefTemp; // temperature scale

    prec SpecGasConst; // specific gas constant

    string DensityName; // density file name
    rinterpol ndDensity; // density rho(T)
    prec RefDensity; // density scale

    string ConductName; // thermal conductivity file name
    rinterpol ndConductivity; // thermal conductivity lambda(T)
    prec RefConductivity; // conductivity scale

    string SpecHeatName; // spec. heat capacity file name
    rinterpol ndSpecHeat; // spec. heat capacity cp(T)
    prec RefSpecHeat; // specific heat scale

    prec LatentMelt; // latent heat for melting
    prec LatentVapor; // latent heat for vaporization

    string IndexName; // refraction index file name
    cinterpol ndIndex; // refraction index (theta)
    cmplx RefIndex; // refraction index scale
    prec AbsCoeff; // absorption coefficient

    rinterpol ndDiffusivity; // diffusivity (theta)
    prec RefDiffusivity; // diffusivity scale

    rinterpol Kelvin; // Kelvin temperature (theta)
    rinterpol ndStefan; // Stefan number (theta)
    prec RefStefan; // Stefan number at Tv

    prec ndLatent; // latent heat ratio cze
    prec ndVapRate; // vaporization velocity cza
};

ostream &operator<<(ostream &ps, TMaterial &mat);
istream &operator>>(istream &ps, TMaterial &src);

extern TMaterial Material;
#endif
