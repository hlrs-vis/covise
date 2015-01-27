/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read IMD checkpoint files from ITAP.                      **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 03.09.2002                                               **
\**************************************************************************/

#include <api/coModule.h>
#include <limits.h>
#include <float.h>
#include <vector>
#include "ReadAstro.h"
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <do/coDoSet.h>
#include <virvo/vvtoolshed.h>

/// Constructor
coReadAstro::coReadAstro(int argc, char *argv[])
    : coModule(argc, argv, "Read files to create a list of points (stars) and scalar parameters.")
    , timestep(0)
    , baseFilename(NULL)
    , currentFilename(NULL)
    , doExec(false)
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "Star location");
    poPoints->setInfo("Star location");

    poSpeed = addOutputPort("Speed", "Vec3", "Star speed");
    poSpeed->setInfo("Star speed");

    poMass = addOutputPort("Mass", "Float", "Star mass");
    poMass->setInfo("Star mass");

    poGalaxy = addOutputPort("Galaxy", "Float", "Galaxy number");
    poGalaxy->setInfo("Galaxy number");

    // Create parameters:
    pbrCheckpointFile = addFileBrowserParam("FilePath", "First file of sequence or single file");
    pbrCheckpointFile->setValue("/data/seminar/data/galaxy/covise/SNAP001", "*");

    pboWarnings = addBooleanParam("Warnings", "Display warnings when reading files");
    pboWarnings->setValue(true);

    pLimit = addInt32Param("LimitTimesteps", "Max. no. of timesteps");
    pLimit->setValue(40);

    pReorder = addBooleanParam("Reorder", "Intersperse stars of first and second galaxy");
    pReorder->setValue(true);

    pboTimestepSets = addBooleanParam("TimestepSets", "Output sets for time steps");
    pboTimestepSets->setValue(true);

    pisNumStarsGalaxy = addInt32Param("NumStarsGalaxy", "Number of stars per galaxy");
    pisNumStarsGalaxy->setValue(24000);

    setExecGracePeriod(0.0);
}

void coReadAstro::param(const char *name, bool /*inMapLoading*/)
{
    if (!strcmp(name, pbrCheckpointFile->getName()))
    {
        timestep = 0;
        if (baseFilename)
            delete[] baseFilename;
        baseFilename = new char[strlen(pbrCheckpointFile->getValue()) + 1];
        strcpy(baseFilename, pbrCheckpointFile->getValue());

        fprintf(stderr, "file: %s\n", pbrCheckpointFile->getValue());

        delete[] currentFilename;
        currentFilename = NULL;
    }
}

/// @return absolute value of a vector
float coReadAstro::absVector(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

/// @return true if warnings are to be displayed
bool coReadAstro::displayWarnings()
{
    return pboWarnings->getValue();
}

bool coReadAstro::readArray(FILE *fp, float **data, int numElems, bool reorder)
{
    *data = new float[numElems];
    if (reorder)
    {
        for (int i = 0; i < numElems; i++)
        {
            int k = (i < numElems / 2) ? i * 2 : (i - numElems / 2) * 2 + 1;
            if (fscanf(fp, "%f\n", &(*data)[k]) != 1)
                return false;
        }
    }
    else
    {
        for (int i = 0; i < numElems; i++)
        {
            if (fscanf(fp, "%f\n", &(*data)[i]) != 1)
                return false;
        }
    }
    return true;
}

float coReadAstro::idle()
{
    if (doExec)
    {
        usleep(100000);
        selfExec();
        return .0f;
    }
    else
    {
        return -1.f;
    }
}

/// Compute routine: load checkpoint file
int coReadAstro::compute(const char *)
{
    bool asSet = pboTimestepSets->getValue();

    // Open first checkpoint file:
    if (!baseFilename)
    {
        baseFilename = new char[strlen(pbrCheckpointFile->getValue()) + 1];
        strcpy(baseFilename, pbrCheckpointFile->getValue());
    }

    FILE *testfp = fopen(baseFilename, "rb");
    if (!testfp)
    {
        sendError("Checkpoint file %s not found.", baseFilename);
        return STOP_PIPELINE;
    }
    else
    {
        fclose(testfp);
    }

    // Create temporary filename that can be modified to increase:
    if (!currentFilename)
    {
        currentFilename = new char[strlen(baseFilename) + 1];
        strcpy(currentFilename, baseFilename);
    }

    int numStarsPerGalaxy = pisNumStarsGalaxy->getValue();

    std::vector<coDistributedObject *> points;
    std::vector<coDistributedObject *> speeds;
    std::vector<coDistributedObject *> masses;
    std::vector<coDistributedObject *> galaxies;

    // Read time steps one by one:

    timestep = 0;
    FILE *fp = fopen(currentFilename, "rb");
    while (fp)
    {
        int numStars = 0;
        int dummy1, dummy2;
        int dim = 0;
        double time;

        // read header
        if (fscanf(fp, "%d %d %d\n", &numStars, &dummy1, &dummy2) != 3)
        {
            cerr << "ReadAstro::compute: fscanf1 failed" << endl;
        }
        if (fscanf(fp, "%d\n", &dim) != 1)
        {
            cerr << "ReadAstro::compute: fscanf2 failed" << endl;
        }
        if (fscanf(fp, "%lf\n", &time) != 1)
        {
            cerr << "ReadAstro::compute: fscanf3 failed" << endl;
        }

        // read the arrays
        float *mass, *x, *y, *z, *vx, *vy, *vz;
        readArray(fp, &mass, numStars, pReorder->getValue());
        readArray(fp, &x, numStars, pReorder->getValue());
        readArray(fp, &y, numStars, pReorder->getValue());
        readArray(fp, &z, numStars, pReorder->getValue());
        readArray(fp, &vx, numStars, pReorder->getValue());
        readArray(fp, &vy, numStars, pReorder->getValue());
        readArray(fp, &vz, numStars, pReorder->getValue());
        float *galaxy = new float[numStars];
        if (pReorder->getValue())
        {
            for (int i = 0; i < numStars; i++)
                galaxy[i] = float(i % 2);
        }
        else
        {
            for (int i = 0; i < numStars; i++)
                galaxy[i] = float(i / numStarsPerGalaxy);
        }

        char name[1024];
        snprintf(name, sizeof(name), "%s_%d", poPoints->getObjName(), timestep);
        coDoPoints *doPoints = new coDoPoints(name, numStars, x, y, z);
        snprintf(name, sizeof(name), "%s_%d", poSpeed->getObjName(), timestep);
        coDoVec3 *doSpeed = new coDoVec3(name, numStars, vx, vy, vz);
        snprintf(name, sizeof(name), "%s_%d", poMass->getObjName(), timestep);
        coDoFloat *doMass = new coDoFloat(name, numStars, mass);
        snprintf(name, sizeof(name), "%s_%d", poGalaxy->getObjName(), timestep);
        coDoFloat *doGalaxy = new coDoFloat(name, numStars, galaxy);

        points.push_back(doPoints);
        speeds.push_back(doSpeed);
        masses.push_back(doMass);
        galaxies.push_back(doGalaxy);
        if (!asSet)
        {
            points.push_back(NULL);
            speeds.push_back(NULL);
            masses.push_back(NULL);
            galaxies.push_back(NULL);

            // Create set objects:
            coDoSet *setPoints = new coDoSet(poPoints->getObjName(), &points[0]);
            coDoSet *setSpeed = new coDoSet(poSpeed->getObjName(), &speeds[0]);
            coDoSet *setMass = new coDoSet(poMass->getObjName(), &masses[0]);
            coDoSet *setGalaxy = new coDoSet(poGalaxy->getObjName(), &galaxies[0]);

            // Now the arrays can be cleared:
            points.clear();
            speeds.clear();
            masses.clear();
            galaxies.clear();

            // Assign sets to output ports:
            poPoints->setCurrentObject(setPoints);
            poSpeed->setCurrentObject(setSpeed);
            poMass->setCurrentObject(setMass);
            poGalaxy->setCurrentObject(setGalaxy);
        }

        fclose(fp);

        timestep++;
        fprintf(stderr, "Loaded timestep %d\n", timestep);
        if (timestep >= pLimit->getValue())
            break;

        // Process next time step:
        if (!vvToolshed::increaseFilename(currentFilename))
        {
            if (!asSet)
            {
                delete[] currentFilename;
                currentFilename = NULL;
                timestep = 0;
            }
            break;
        }
        fp = fopen(currentFilename, "rb");
        if (!fp)
        {
            delete[] currentFilename;
            currentFilename = NULL;
            if (!asSet)
                timestep = 0;
        }

        if (!asSet)
        {
            if (timestep > 0)
            {
                doExec = true;
                return CONTINUE_PIPELINE;
            }
            else
            {
                doExec = false;
                return STOP_PIPELINE;
            }
        }
    }
    doExec = false;

    if (asSet)
    {
        delete[] currentFilename;
        currentFilename = NULL;
    }

    if (timestep == 0)
    {
        sendError("No stars loaded.");
        return STOP_PIPELINE;
    }

    if (asSet)
    {
        points.push_back(NULL);
        speeds.push_back(NULL);
        masses.push_back(NULL);
        galaxies.push_back(NULL);

        // data has been loaded and can now be converted to sets

        // Create set objects:
        coDoSet *setPoints = new coDoSet(poPoints->getObjName(), &points[0]);
        coDoSet *setSpeed = new coDoSet(poSpeed->getObjName(), &speeds[0]);
        coDoSet *setMass = new coDoSet(poMass->getObjName(), &masses[0]);
        coDoSet *setGalaxy = new coDoSet(poGalaxy->getObjName(), &galaxies[0]);

        // Now the arrays can be cleared:
        points.clear();
        speeds.clear();
        masses.clear();
        galaxies.clear();

        // Set timestep attribute:
        if (timestep > 1)
        {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%d %d", 0, timestep - 1);
            setPoints->addAttribute("TIMESTEP", buf);
            setSpeed->addAttribute("TIMESTEP", buf);
            setMass->addAttribute("TIMESTEP", buf);
            setGalaxy->addAttribute("TIMESTEP", buf);
        }

        // Assign sets to output ports:
        poPoints->setCurrentObject(setPoints);
        poSpeed->setCurrentObject(setSpeed);
        poMass->setCurrentObject(setMass);
        poGalaxy->setCurrentObject(setGalaxy);

        sendInfo("Timesteps loaded: %d", timestep);
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadAstro)
