/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          simul.h  -  simulation settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __SIMUL_H_

#define __SIMUL_H_

#include <string>

#include "classext.h"

// ***************************************************************************
// simulation settings
// ***************************************************************************

class TSimulation
{
public:
    TSimulation()
    {
        Reset();
    }
    TSimulation(const TSimulation &);

    void Save(int); // save settings
    void Read(int, float vers = 1.0); // read settings
    void Reset(); // reset values
    void Update(); // update values
    TSimulation GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form

    int Initialize(); // initialize simulation (once)
    int NextTimeStep(); // initialize next time step
    void RestoreLastStep(); // restore last time step

    string ProfileName; // name of used profile
    bool bChanged; // change flag

    prec ndTime; // current time
    prec ndPulseTime; // time since begin of pulse
    prec ndTimeEnd; // simulation stop
    int iTimeSteps; // number of time steps
    prec ndDeltat; // current time step
    prec ndDeltatOn; // on time step
    prec ndDeltatOff; // off time step

    bool bMultReflec; // flag for multiple reflections
    bool bSpecReflec; // flag for specular reflections
    int iReflection; // kind of reflection
    enum Reflection
    {
        constant = 0,
        unpolarized = 1,
        polarized = 2
    };
    prec ndRefRadius; // reflection radius (cs2)
    int iAveRays; // average ray number

    bool bSurfVapor; // flag surface vaporization
    bool bAutoSave; // flag auto save
    int iAutoSaveSteps; // number steps between auto save
    prec ndAutoSaveTime; // time delay between auto save
    string AutoSaveName; // name for auto save
    bool bFollowLaser; // flag for moving grid with laser
    bool bHeatConduct; // heat conduction simulation
    prec ndHeatingDelay; // delay for heating w/o conduction
    prec ndCoolingDelay; // delay for cooling w/o conduction

    prec ndMaxSurfMove; // maximum displacement of surface node
    prec ndMaxGridMove; // max. displacement before grid update
    prec ndMaxZMove; // max. displacement z-direction
    prec ndMaxXYMove; // max. displacement x,y-direction
    prec ndTempTol; // tolerance for temperature iteration
    prec ndTempOff; // temperature limit for on/off timestep
    prec ndMinTemp; // negligable temperature
    prec ndMinVapVelo; // negligable vaporization velocity
    prec ndMinPower; // negligable power

    int iStatus; // status flag for simulation
    enum Status
    {
        new_grid = 0,
        loaded = 1,
        simulated = 2
    };
};

ostream &operator<<(ostream &, TSimulation &);
istream &operator>>(istream &, TSimulation &);

extern TSimulation Simulation, SimulationOld;
#endif
