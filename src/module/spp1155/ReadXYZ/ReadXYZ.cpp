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

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <api/coModule.h>
#include <alg/coChemicalElement.h>
#include <limits.h>
#include <float.h>
#include <cassert>
#include "ReadXYZ.h"

/// Constructor
coReadXYZ::coReadXYZ(int argc, char *argv[])
    : coModule(argc, argv, "Read XYZ files containing lists of atom positions and their element types for several time steps.")
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "Atom location");
    poTypes = addOutputPort("Element", "Int", "Atom element type");
    poBounds = addOutputPort("Bounds", "Points", "Boundary atoms");

    // Create parameters:
    pbrFilename = addFileBrowserParam("Filename", "XYZ file");
    pbrFilename->setValue("data/", "*.xyz;*.XYZ/*.*");

    pboReadBounds = addBooleanParam("ReadBounds", "Interpret first 8 atoms as bounding box");
    pboReadBounds->setValue(true);

    pLimitTimesteps = addInt32Param("LimitTimestep", "Maximum number of timesteps to read (0 = all)");
    pLimitTimesteps->setValue(0);

}

/// Compute routine: load checkpoint file
int coReadXYZ::compute(const char *)
{
    bool readbbox = pboReadBounds->getValue();
    // Open first checkpoint file:
    const char *path = pbrFilename->getValue();
    int timestepLimit = pLimitTimesteps->getValue();

    float minmax[2][3] = {
        { FLT_MAX, FLT_MAX, FLT_MAX },
        { -FLT_MAX, -FLT_MAX, -FLT_MAX }
    };

    std::vector<coDoPoints *> points;
    std::vector<coDoInt *> types;
    // Read time steps one by one:
    int timestep = 0;
    int totalAtoms = 0;

    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        sendError("failed to open XYZ file %s for reading.", path);
        return STOP_PIPELINE;
    }
    char lineData[5000];
    fgets(lineData, 5000, fp);
    while (!feof(fp))
    {
        if (timestepLimit > 0 && timestep >= timestepLimit)
            break;

        int dummyNumAtoms = 0;
        if (sscanf(lineData, "%d", &dummyNumAtoms) != 1)
        {
            if(timestep == 0)
            { 
            sendError("Failed to read number of atoms.");
            fclose(fp);
            return STOP_PIPELINE;
            }
            break;
        }

        char line[1024];
        if (!fgets(line, sizeof(line), fp))
        {
            sendError("Failed to read description for timestep %d", timestep);
            fclose(fp);
            return STOP_PIPELINE;
        }

        fprintf(stderr, "reading timestep %d with %d atoms\n", timestep, dummyNumAtoms);

        if (timestep == 0 && readbbox)
        {
            dummyNumAtoms -= 8;
            for (int i = 0; i < 8; i++)
            {
                char buf[1024];
                float x[3];
                fgets(lineData, 5000, fp);
                if (sscanf(lineData, "%3s %f %f %f\n", buf, &x[0], &x[1], &x[2]) != 4)
                {
                    sendError("Failed to read data for bounds.");
                    fclose(fp);
                    return STOP_PIPELINE;
                }
                for (int j = 0; j < 3; j++)
                {
                    if (x[j] < minmax[0][j])
                        minmax[0][j] = x[j];
                    if (x[j] > minmax[1][j])
                        minmax[1][j] = x[j];
                }
            }
            float *x, *y, *z;
            coDoPoints *doBounds = new coDoPoints(poBounds->getObjName(), 2);
            doBounds->getAddresses(&x, &y, &z);
            for (int i = 0; i < 2; i++)
            {
                x[i] = minmax[i][0];
                y[i] = minmax[i][1];
                z[i] = minmax[i][2];
            }
            poBounds->setCurrentObject(doBounds);
        }

        std::vector<int> vt;
        std::vector<float> vx, vy, vz;
        do
        {
            //fpos_t pos;
            //fgetpos(fp, &pos);
            char buf[1024];
            float x, y, z;

            fgets(lineData, 5000, fp);
            int n = sscanf(lineData, "%s %f %f %f\n", buf, &x, &y, &z);
            if (n != 4)
            {
                if (n <= 2)
                {
                    //fsetpos(fp, &pos);
                    break;
                }
                sendError("Failed to read data for atom.");
                fclose(fp);
                return STOP_PIPELINE;
            }

            vx.push_back(x);
            vy.push_back(y);
            vz.push_back(z);

            auto elem = coAtomInfo::instance()->idMap.find(buf);
            if (elem == coAtomInfo::instance()->idMap.end())
                vt.push_back(0);
            else
                vt.push_back(elem->second);
        } while (!feof(fp));
        int numAtoms = (int)vt.size();
        totalAtoms += numAtoms;

        char name[1024];
        snprintf(name, sizeof(name), "%s_%d", poTypes->getObjName(), timestep);
        coDoInt *doTypes = new coDoInt(name, numAtoms, &vt[0]);

        snprintf(name, sizeof(name), "%s_%d", poPoints->getObjName(), timestep);
        coDoPoints *doPoints = new coDoPoints(name, numAtoms, &vx[0], &vy[0], &vz[0]);

        points.push_back(doPoints);
        types.push_back(doTypes);

        // Process next time step:
        timestep++;
    }
    fclose(fp);

    if (timestep == 0)
    {
        sendError("No atoms loaded.");
        return STOP_PIPELINE;
    }

    // Create set objects:
    coDoSet *setPoints = new coDoSet(poPoints->getObjName(), (int)points.size(), (coDistributedObject **)&points[0]);
    coDoSet *setTypes = new coDoSet(poTypes->getObjName(), (int)types.size(), (coDistributedObject **)&types[0]);
    // Now the arrays can be cleared:
    points.clear();
    types.clear();

    // Set timestep attribute:
    if (timestep > 1)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%d %d", 0, timestep - 1);
        setPoints->addAttribute("TIMESTEP", buf);
        setTypes->addAttribute("TIMESTEP", buf);
    }

    // Assign sets to output ports:
    poPoints->setCurrentObject(setPoints);
    poTypes->setCurrentObject(setTypes);

    sendInfo("Timesteps loaded: %d (%d atoms)", timestep, totalAtoms);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadXYZ)
