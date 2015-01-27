/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "trans3DInterface.h"
#include "coviseInterface.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <string>
#ifdef __sgi
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#else
#include <cstdlib>
#include <cctype>
#include <ctime>
#endif

#include "main.h"
#include "fortran.h"
#include "raytrace.h"
#include "classext.h"
#include "grid.h"
#include "laser.h"
#include "material.h"
#include "simul.h"
#include "solid.h"
#include "solve.h"

extern void init_program(const char *filename);
extern int CalculateAblation();
extern void WriteGlobalVariables();
extern bool ShouldAbort();

trans3DInterface::trans3DInterface()
{
}

trans3DInterface::~trans3DInterface()
{
}

void trans3DInterface::init(const char *filename)
{
    init_program(filename);
}

void trans3DInterface::getGridSize(int &xDim, int &yDim, int &zDim)
{
    xDim = Grid.iVolNodes + 2;
    yDim = Grid.jVolNodes + 2;
    zDim = Grid.kVolNodes;
}

void trans3DInterface::getValues(int i, int j, int k, float *xc, float *yc, float *zc, float *t, float *q)
{
    *xc = Solid(i, j, k + 1).ndNode.x;
    *yc = Solid(i, j, k + 1).ndNode.y;
    *zc = Solid(i, j, k + 1).ndNode.z;
    // Umrechnung von Kirchhoff nach Kelvin
    *t = Material.RoomTemp + Material.VaporTemp * Solid(i, j, k + 1).ndTemp;
    *q = Solid(i, j, k + 1).ndHeat;
}

int trans3DInterface::Calculate(int numSteps)
{
    char buf[1000];
    //int   i;
    int rv;
    int ns = 0;

    try
    {
        while ((Simulation.iTimeSteps > 0 && nstep < Simulation.iTimeSteps) || (Simulation.iTimeSteps == 0 && Simulation.ndTime / Simulation.ndTimeEnd - 1.0 < -1.0e-12))
        {
            if (ns >= numSteps)
                return 1; // continue next time...
            ns++;
            nstep++;
            do
            {
                if ((rv = Simulation.NextTimeStep()) < 0)
                    return rv;
                //        WriteGlobalVariables();

                //
                Grid.iGridMove = TGrid::no_update;

                //Nur dann wird die Laserabsorption garantiert neu berechnet
                //(sonst
                //kann, um Zeit zu sparen, einfach der Wert des letzten Zeitschritts neu
                //skaliert
                //werden). Ausserdem habe ich in laser.cpp eine Aenderung eingefuegt, die
                //bewirkt,
                //dass die Absorption beim Verschieben des Lasers immer neu berechnet
                //wird.

                if ((rv = Solid.Solve(ress, resd, dtmax)) < 0)
                    return rv;
                Grid.UpdateGrid();
            } while (Grid.iGridMove == TGrid::reduced_step);

            dimtime = Simulation.ndPulseTime * RefTime;
            sprintf(buf, "%10.6le %4i  %10.3le %10.3le %10.3le %10.3le %7i %7i",
                    dimtime, nstep, dtmax, Solid.ndMaxTemp,
                    Laser.ndAvePower, Solid.ndVapRate, Laser.iRays, Laser.iBadRays);

            covise.info(buf);
        }
    }
    catch (TException &ex)
    {
        Simulation.RestoreLastStep();
        ex.Display();
        return -1;
    }
    catch (...)
    {
        Simulation.RestoreLastStep();
        ErrorFunction("Unbekannter Programmfehler");
        return -1;
    }
    nstep = 0;
    return 0;
}

int trans3DInterface::executeScript()
{
    return script.execute();
}

int trans3DInterface::initCalculation()
{
    int rv;
    covise.info("initializing calculation...");
    try
    {
        if ((rv = Simulation.Initialize()) < 0)
            return rv;

        srand((unsigned)time(NULL));
        covise.info("    Time     Step#    Max dT      Max T     AvePower"
                    "   RemRate  # Rays Bad Rays");

        nstep = 0;
    }
    catch (TException &ex)
    {
        Simulation.RestoreLastStep();
        ex.Display();
        return -1;
    }
    catch (...)
    {
        Simulation.RestoreLastStep();
        ErrorFunction("Unbekannter Programmfehler");
        return -1;
    }

    return 0;
}

void trans3DInterface::setIntensity(float factor)
{
    Laser.PowerFactor = factor;
}

void trans3DInterface::setLaserPos(float x, float y, float z)
{
    Laser.ndInitialPos.x = x;
    Laser.ndInitialPos.y = y;
    Laser.ndInitialPos.z = z;
}

void trans3DInterface::getLaserPos(float &x, float &y, float &z)
{
    x = Laser.ndInitialPos.x;
    y = Laser.ndInitialPos.y;
    z = Laser.ndInitialPos.z;
}

void trans3DInterface::setRadius(float factor)
{
    Laser.RadiusFactor = factor;
}

trans3DInterface trans3D;
