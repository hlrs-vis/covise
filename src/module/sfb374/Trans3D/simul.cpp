/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          simul.cpp  -  simulation settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "simul.h"
#include "fortran.h"
#include "solid.h"
#include "laser.h"
#include "material.h"
#include "grid.h"
#include "main.h"

TSimulation Simulation, SimulationOld;

// ***************************************************************************
// class for simulation settings
// ***************************************************************************

// copy constructor

TSimulation::TSimulation(const TSimulation &src)
{
    ProfileName = src.ProfileName;
    bChanged = src.bChanged;
    ndTime = src.ndTime;
    ndPulseTime = src.ndPulseTime;
    ndTimeEnd = src.ndTimeEnd;
    iTimeSteps = src.iTimeSteps;
    ndDeltat = src.ndDeltat;
    ndDeltatOn = src.ndDeltatOn;
    ndDeltatOff = src.ndDeltatOff;
    bMultReflec = src.bMultReflec;
    bSpecReflec = src.bSpecReflec;
    iReflection = src.iReflection;
    ndRefRadius = src.ndRefRadius;
    iAveRays = src.iAveRays;
    bSurfVapor = src.bSurfVapor;
    bAutoSave = src.bAutoSave;
    iAutoSaveSteps = src.iAutoSaveSteps;
    ndAutoSaveTime = src.ndAutoSaveTime;
    AutoSaveName = src.AutoSaveName;
    bFollowLaser = src.bFollowLaser;
    bHeatConduct = src.bHeatConduct;
    ndHeatingDelay = src.ndHeatingDelay;
    ndCoolingDelay = src.ndCoolingDelay;
    ndMaxSurfMove = src.ndMaxSurfMove;
    ndMaxGridMove = src.ndMaxGridMove;
    ndMaxZMove = src.ndMaxZMove;
    ndMaxXYMove = src.ndMaxXYMove;
    ndTempTol = src.ndTempTol;
    ndTempOff = src.ndTempOff;
    ndMinTemp = src.ndMinTemp;
    ndMinVapVelo = src.ndMinVapVelo;
    ndMinPower = src.ndMinPower;
    iStatus = src.iStatus;
}

// reset variables

void TSimulation::Reset()
{
    ProfileName.erase();
    bChanged = false;
    ndTime = 0;
    ndTimeEnd = 2.5e-3;
    iTimeSteps = 100;
    ndDeltatOn = 1e-5;
    ndDeltatOff = 1e-5;
    bMultReflec = false;
    bSpecReflec = false;
    iReflection = constant;
    ndRefRadius = 1e-10;
    iAveRays = 1;
    bSurfVapor = true;
    bAutoSave = false;
    iAutoSaveSteps = 100;
    ndAutoSaveTime = 2.5e-3;
    AutoSaveName.erase();
    bFollowLaser = false;
    bHeatConduct = true;
    ndHeatingDelay = 0;
    ndCoolingDelay = 0;
    ndMaxSurfMove = 1e-4;
    ndMaxGridMove = 1e-3;
    ndMaxZMove = 0.1;
    ndMaxXYMove = 0.1;
    ndTempTol = 1e-4;
    ndTempOff = 0.8;
    ndMinTemp = 0.05;
    ndMinVapVelo = 1e-6;
    iStatus = new_grid;
}

// ***************************************************************************
// save settings
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TSimulation &s)
{
    TSimulation src = s.GetDimensional();

    ps << "simulation settings:" << endl;
    ps << endl;
    ps << "profile name:\t" << src.ProfileName << endl;
    ps << "current time [s]:\t" << src.ndTime << endl;
    ps << "final time [s]:\t" << src.ndTimeEnd << endl;
    ps << "number of time steps:\t" << src.iTimeSteps << endl;
    ps << "time step length [s]:\t" << src.ndDeltat << endl;
    ps << "time step length pulse on [s]:\t" << src.ndDeltatOn << endl;
    ps << "time step length pulse off [s]:\t" << src.ndDeltatOff << endl;
    ps << "multiple reflections:\t" << src.bMultReflec << endl;
    ps << "specular reflection:\t" << src.bSpecReflec << endl;
    ps << "type of reflection:\t" << src.iReflection << endl;
    ps << "diffuse radius [m]:\t" << src.ndRefRadius << endl;
    ps << "average ray number:\t" << src.iAveRays << endl;
    ps << "surface evaporation:\t" << src.bSurfVapor << endl;
    ps << "auto save:\t" << src.bAutoSave << endl;
    ps << "steps until auto save:\t" << src.iAutoSaveSteps << endl;
    ps << "time until auto save [s]:\t" << src.ndAutoSaveTime << endl;
    ps << "name of auto save file:\t" << src.AutoSaveName << endl;
    ps << "grid motion with laser:\t" << src.bFollowLaser << endl;
    ps << "heat conduction:\t" << src.bHeatConduct << endl;
    ps << "heatind delay time [s]:\t" << src.ndHeatingDelay << endl;
    ps << "cooling delay time [s]:\t" << src.ndCoolingDelay << endl;
    ps << "max surface nodal motion [m]:\t" << src.ndMaxSurfMove << endl;
    ps << "max nodal motion [m]\t" << src.ndMaxGridMove << endl;
    ps << "max z nodal motion [m]\t" << src.ndMaxZMove << endl;
    ps << "max xy nodal motion [m]\t" << src.ndMaxXYMove << endl;
    ps << "temperature tolerance [K]\t" << src.ndTempTol << endl;
    ps << "temp limit for pulse off [K]\t" << src.ndTempOff << endl;
    ps << "negligable temperature [K]\t" << src.ndMinTemp << endl;
    ps << "negligable ablation velocity [m/s]:\t" << src.ndMinVapVelo << endl;
    ps << "negligable power density:\t" << src.ndMinPower << endl;
    ps << "simulation status:\t" << src.iStatus << endl;
    return ps << endl;
}

// ***************************************************************************
// read settings
//
// output:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

istream &operator>>(istream &ps, TSimulation &src)
{
    src.bChanged = true;
    if (!CheckHeader(ps, "simulation settings:"))
        return ps;
    src.bChanged = false;
    ps >> tab;
    getline(ps, src.ProfileName);
    ps >> tab >> src.ndTime;
    ps >> tab >> src.ndTimeEnd;
    ps >> tab >> src.iTimeSteps;
    ps >> tab >> src.ndDeltat;
    ps >> tab >> src.ndDeltatOn;
    ps >> tab >> src.ndDeltatOff;
    ps >> tab >> src.bMultReflec;
    ps >> tab >> src.bSpecReflec;
    ps >> tab >> src.iReflection;
    ps >> tab >> src.ndRefRadius;
    ps >> tab >> src.iAveRays;
    ps >> tab >> src.bSurfVapor;
    ps >> tab >> src.bAutoSave;
    ps >> tab >> src.iAutoSaveSteps;
    ps >> tab >> src.ndAutoSaveTime >> tab;
    getline(ps, src.AutoSaveName);
    ps >> tab >> src.bFollowLaser;
    ps >> tab >> src.bHeatConduct;
    ps >> tab >> src.ndHeatingDelay;
    ps >> tab >> src.ndCoolingDelay;
    ps >> tab >> src.ndMaxSurfMove;
    ps >> tab >> src.ndMaxGridMove;
    ps >> tab >> src.ndMaxZMove;
    ps >> tab >> src.ndMaxXYMove;
    ps >> tab >> src.ndTempTol;
    ps >> tab >> src.ndTempOff;
    ps >> tab >> src.ndMinTemp;
    ps >> tab >> src.ndMinVapVelo;
    ps >> tab >> src.ndMinPower;
    ps >> tab >> src.iStatus >> endl >> endl;
    src.MakeNonDimensional();
    src.Update();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
//
// input:   RefLength, RefTime, RefVelocity, RefTemp
//
// ***************************************************************************

TSimulation TSimulation::GetDimensional()
{
    TSimulation s = *this;

    s.ndTime *= RefTime;
    s.ndPulseTime *= RefTime;
    s.ndTimeEnd *= RefTime;
    s.ndDeltat *= RefTime;
    s.ndDeltatOn *= RefTime;
    s.ndDeltatOff *= RefTime;
    s.ndRefRadius *= RefLength;
    s.ndAutoSaveTime *= RefTime;
    s.ndHeatingDelay *= RefTime;
    s.ndCoolingDelay *= RefTime;
    s.ndMaxSurfMove *= RefLength;
    s.ndMaxGridMove *= RefLength;
    s.ndMaxZMove *= RefLength;
    s.ndMaxXYMove *= RefLength;
    s.ndMinVapVelo *= RefVelocity;
    s.ndTempTol = Material.GetKelvin(s.ndTempTol);
    s.ndTempOff = Material.GetKelvin(s.ndTempOff);
    s.ndMinTemp = Material.GetKelvin(s.ndMinTemp);
    return s;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// output:    RefLength, RefTime, RefVelocity, RefTemp
//
// ***************************************************************************

void TSimulation::MakeNonDimensional()
{
    ndTime /= RefTime;
    ndPulseTime /= RefTime;
    ndTimeEnd /= RefTime;
    ndDeltat /= RefTime;
    ndDeltatOn /= RefTime;
    ndDeltatOff /= RefTime;
    ndRefRadius /= RefLength;
    ndAutoSaveTime /= RefTime;
    ndHeatingDelay /= RefTime;
    ndCoolingDelay /= RefTime;
    ndMaxSurfMove /= RefLength;
    ndMaxGridMove /= RefLength;
    ndMaxZMove /= RefLength;
    ndMaxXYMove /= RefLength;
    ndMinVapVelo /= RefVelocity;
    ndTempTol = Material.GetKirchhoff(ndTempTol);
    ndTempOff = Material.GetKirchhoff(ndTempOff);
    ndMinTemp = Material.GetKirchhoff(ndMinTemp);
}

// ***************************************************************************
// update dependent variables
//
// output:      ndPulseTime, ndMinPower
// input:       iAveRays, ndTime, ndPulselength, ndPulseOn, Grid.iVolNodes
//
// call only after Laser and Grid update
//
// ***************************************************************************

void TSimulation::Update()
{
    ndPulseTime = ndTime - int(ndTime / Laser.ndPulselength) * Laser.ndPulselength;
    //  if(iStatus!=simulated)  // only for new setup
    ndMinPower = 1e-3 * PI_OVER_TWO * Laser.ndPulselength / (Laser.ndPulseOn * max((Grid.iIniNodes - 1) * (Grid.jIniNodes - 1), iAveRays));
}

// ***************************************************************************
// save settings (temporary)
// ***************************************************************************

void TSimulation::Save(int unit)
{
    unitwriteln(unit, "\nSimulation:");
    unitwrite(unit, "Profilname =\t");
    unitwriteln(unit, ProfileName.c_str());
    unitwrite(unit, iTimeSteps, "Rechenschritte =\t%i\n");
    unitwrite(unit, ndTimeEnd * RefTime, "Zeit Rechenende =\t%le\n");
    unitwrite(unit, iAveRays, "Strahlen =\t%i\n");
    unitwrite(unit, bMultReflec, "Vielfachreflektion =\t%i\n");
    unitwrite(unit, bSpecReflec, "Reflektionstyp =\t%i\n");
    unitwrite(unit, iReflection, "Reflektionsgesetz =\t%i\n");
    unitwrite(unit, int(0), "Temperaturabhängigkeit =\t%i\n");
    unitwrite(unit, ndRefRadius * RefLength, "cs2 =\t%le\n");
    unitwrite(unit, ndMaxSurfMove * RefLength, "max. Verschiebung OF-Knoten =\t%le\n");
    unitwrite(unit, ndTempTol, "Temperaturgenauigkeit =\t%le\n");
    unitwrite(unit, ndMinVapVelo * RefVelocity, "minimale Ablationsgeschwindigkeit =\t%le\n");
    unitwrite(unit, ndMaxGridMove * RefLength, "max. Verschiebung bis Update=\t%le\n");
    unitwrite(unit, ndTempOff, "Temperaturschwelle an/aus =\t%le\n");
    unitwrite(unit, ndMinTemp, "Mindestrechentemperatur =\t%le\n");
    unitwrite(unit, ndMaxZMove * RefLength, "max. Z-Verschiebung =\t%le\n");
    unitwrite(unit, ndMaxXYMove * RefLength, "max. XY-Verschiebung =\t%le\n");
    unitwrite(unit, ndDeltatOn * RefTime, "Zeitschritt an =\t%le\n");
    unitwrite(unit, ndDeltatOff * RefTime, "Zeitschritt aus =\t%le\n");
    unitwrite(unit, bHeatConduct, "Wärmeleitung =\t%i\n");
    unitwrite(unit, ndHeatingDelay * RefTime, "Zeitverzoegerung =\t%le\n");
    unitwrite(unit, ndCoolingDelay * RefTime, "ZeitverzoegerungEnde =\t%le\n");
    unitwrite(unit, bSurfVapor, "Oberflaechenverdampfung =\t%i\n");
    unitwrite(unit, bAutoSave, "Autospeicherung =\t%i\n");
    unitwrite(unit, iAutoSaveSteps, "AutoSchritte =\t%i\n");
    unitwrite(unit, ndAutoSaveTime * RefTime, "AutoZeit =\t%le\n");
    unitwrite(unit, "AutoDateiname =\t");
    unitwriteln(unit, AutoSaveName.c_str());
    unitwrite(unit, bFollowLaser, "Gitter folgt Laser =\t%i\n");
}

// ***************************************************************************
// read settings (temporary)
// ***************************************************************************

void TSimulation::Read(int unit, float vers)
{
    char buffer[100];

    unitreadln(unit, buffer);
    unitreadln(unit, buffer);
    ProfileName = unitreadln(unit, buffer);
    iTimeSteps = readint(unit);
    ndTimeEnd = readreal(unit) / RefTime;
    iAveRays = readint(unit);
    bMultReflec = readbool(unit);
    bSpecReflec = readbool(unit);
    iReflection = readint(unit);
    readint(unit); // idmatrl
    ndRefRadius = readreal(unit) / RefLength;
    ndMaxSurfMove = readreal(unit) / RefLength;
    ndTempTol = readreal(unit);
    ndMinVapVelo = readreal(unit) / RefVelocity;
    ndMaxGridMove = readreal(unit) / RefLength;
    ndTempOff = readreal(unit);
    ndMinTemp = readreal(unit);
    ndMaxZMove = readreal(unit) / RefLength;
    ndMaxXYMove = readreal(unit) / RefLength;
    ndDeltatOn = readreal(unit) / RefTime;
    ndDeltatOff = readreal(unit) / RefTime;
    bHeatConduct = readbool(unit);
    ndHeatingDelay = readreal(unit) / RefTime;
    ndCoolingDelay = readreal(unit) / RefTime;
    bSurfVapor = readbool(unit);
    bAutoSave = readbool(unit);
    iAutoSaveSteps = readint(unit);
    ndAutoSaveTime = readreal(unit) / RefTime;
    AutoSaveName = unitreadln(unit, buffer);
    bFollowLaser = readbool(unit);

    Update();
}

// ***************************************************************************
// initialize simulation
// ***************************************************************************

int TSimulation::Initialize()
{
    int rv = 0;

    UpdateVariables(); // probably not necessary
    if (iStatus == new_grid)
        LaserOld.ndRelPower = 0; // power of last time step for new simulation
    return rv;
}

// ***************************************************************************
// restore last time step
// ***************************************************************************

void TSimulation::RestoreLastStep(void)
{
    // to do: restore only relevant data
    Laser = LaserOld; // restore old laser settings
    Grid = GridOld; // restore old grid settings
    *this = SimulationOld; // restore old simulation settings
    Solid = SolidOld; // restore old solid data
    SolSurf = SolSurfOld; // restore old solid data
}

// ***************************************************************************
// initialize next time step
//
// output:  ndTime, ndPulseTime, Laser: ndPosition, ndAvePower, ndRelPower,
//          Solid: ndNode
// input:   calculated values from last time step
//
// ***************************************************************************

int TSimulation::NextTimeStep(void)
{
    if (Grid.iGridMove == TGrid::reduced_step) // decrease time step and calc again
    {
        prec tmp;

        WarningFunction("warning: node movement too fast, dz=%7.3lf,"
                        " saitmax=%5.3lf, etatmax=%5.3lf\n"
                        "decrease dt and/or increase smin; trying dt=%10.4lf",
                        Solid.ndMaxDSai.z * RefLength,
                        Solid.ndMaxDSai.x * RefLength, Solid.ndMaxDSai.y * RefLength,
                        ndDeltat * RefTime);
        tmp = ndDeltat / 2; // try half of previous time step

        RestoreLastStep(); // restoring last time step

        ndDeltat = tmp;
        ndTime += ndDeltat;
        ndPulseTime += ndDeltat;
        if (ndPulseTime >= Laser.ndPulselength)
            ndPulseTime -= Laser.ndPulselength; // time since start of pulse
    }
    else // next time step
    {
        if (ndPulseTime > Laser.ndPulseOn && Solid.ndMaxTemp < 0.1 * ndTempOff)
        { // material cold => advance to next pulse
            ndDeltat = ndDeltatOn * 1.0e-6; // pick small time step
            ndTime = (int(ndTime / Laser.ndPulselength) + 1) * Laser.ndPulselength;
            ndPulseTime = 0;
            Grid.iGridMove = TGrid::updated; // as if grid has been updated
            Solid.ResetTemperature(); // material cold
        }
        else
        {
            if (ndPulseTime > Laser.ndPulseOn && ndPulseTime <= (Laser.ndPulselength - ndDeltatOff) && Solid.ndMaxTemp < ndTempOff && ndDeltatOff > 0) // pulse off
                ndDeltat = ndDeltatOff;
            else
                ndDeltat = ndDeltatOn; // pulse on
            if (ndPulseTime + ndDeltat * 1.1 > Laser.ndPulselength)
                // finish pulse
                ndDeltat = Laser.ndPulselength - ndPulseTime;

            ndTime += ndDeltat;
            ndPulseTime += ndDeltat;
            if (ndPulseTime >= Laser.ndPulselength)
                ndPulseTime -= Laser.ndPulselength; // time since start of pulse
        }
    }

    // to do: save only relevant data
    LaserOld = Laser; // save old laser settings

    Laser.GetPosition(ndTime);
    Laser.GetPower(ndPulseTime); // current power
    Laser.ndAvePower = 0.5 * (Laser.ndRelPower + LaserOld.ndRelPower);

    Grid.UpdateVolumeGrid(); // adding or subtracting of nodes

    GridOld = Grid; // save old grid settings
    SimulationOld = *this; // save old simulation settings
    SolidOld = Solid; // save old solid data
    SolSurfOld = SolSurf; // save old solid data
    return 0;
}
