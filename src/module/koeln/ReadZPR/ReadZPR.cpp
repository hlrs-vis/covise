/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                          (C)2004 RRZK  **
 **                                                                        **
 ** Description: Read CRD files from ZPR.                                  **
 **                                                                        **
 ** Author: Martin Aumueller (aumueller@uni-koeln.de)                      **
 **                                                                        **
 ** Creation Date: 14.12.2004                                              **
\**************************************************************************/

#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <api/coModule.h>
#include <limits.h>
#include <float.h>
#include "ReadZPR.h"

/// Constructor
coReadZPR::coReadZPR(int argc, char *argv[])
    : coModule(argc, argv, "Read ZPR/CRD files containing lists of atom positions and their element types for several time steps.")
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "Particle location");
    poPoints->setInfo("Particle location");

    // Create parameters:
    pbrFilename = addFileBrowserParam("FilePath", "crd file");
    pbrFilename->setValue("src/application/koeln/ReadZPR/testdata.crd", "*.crd");
}

/// Compute routine: load checkpoint file
int coReadZPR::compute(const char *)
{
    // Open first checkpoint file:
    const char *path = pbrFilename->getValue();

    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        sendError("crd file %s not found.", path);
        return STOP_PIPELINE;
    }

    // Read time steps one by one:
    int totalParticles = 0;
    int numTimeSteps = 0;
    vector<coDoPoints *> pointLists;
    const char *poPointName = poPoints->getObjName();

    // Header for each time step
    float time = 0.0;
    char dummy[1024];
    // eat leading newline
    if (fscanf(fp, "%[ \n]", dummy) != 1)
    {
        //fprintf(stderr, "error at header\n");
    }

    while (!feof(fp))
    {
        if (fscanf(fp, "%f%[ \n]", &time, dummy) != 2)
        {
            if (!feof(fp))
            {
                fprintf(stderr, "error at timestep header\n");
            }
        }

        numTimeSteps++;

        vector<float> x_list;
        vector<float> y_list;
        vector<float> z_list;
        //vector<int> particleNumList = new vector<int>;
        //vector<int> particleKindList = new vector<int>;

        // Read the complete timestep
        while (!feof(fp))
        {

            int fieldsRead = fscanf(fp, "%[ ]", dummy);
            //fprintf(stderr, "fields read for space munger: %d\n", fieldsRead);
            fieldsRead = fscanf(fp, "%[\n]", dummy);
            if (fieldsRead == 1 && strlen(dummy) > 1)
            {
                // next timestep begins with a newline
                break;
            }

            int particleNum, particleKind;
            fieldsRead = fscanf(fp, "%d%[ ]%d%[\n]", &particleNum, dummy, &particleKind, dummy);
            if (fieldsRead != 4)
            {
                if (feof(fp))
                {
                    break;
                }
                else
                {
                    fprintf(stderr, "%d fields for particle number and kind read instead of 4: particleNum=%d\n", fieldsRead, particleNum);
                    break;
                }
            }

            float x, y, z;
            fieldsRead = fscanf(fp, "%f%[ ]%f%[ ]%f", &x, dummy, &y, dummy, &z);
            if (fieldsRead != 5)
            {
                if (feof(fp))
                {
                    break;
                }
                else
                {
                    fprintf(stderr, "%d fields for particle position read instead of 5\n", fieldsRead);
                    break;
                }
            }

            //particleNumList.push_back(particleNum);
            //particleKindList.push_back(particleKind);

            x_list.push_back(x);
            y_list.push_back(y);
            z_list.push_back(z);

            totalParticles++;
        }

        char buf[1024];
        sprintf(buf, "%s_%d", poPointName, numTimeSteps - 1);
        pointLists.push_back(new coDoPoints(buf, (int)x_list.size(),
                                            &x_list.front(), &y_list.front(), &z_list.front()));
        x_list.clear();
        y_list.clear();
        z_list.clear();
        //particleNumList.clear();
        //particleKindList.clear();

        if (feof(fp))
        {
            break;
        }
    }
    fclose(fp);

    pointLists.push_back(NULL); // for termination

    if (numTimeSteps == 0)
    {
        sendError("No particles loaded.");
        return STOP_PIPELINE;
    }

    sendInfo("Timesteps loaded: %d (%d atoms)", numTimeSteps, totalParticles);

    // combine point lists into output set
    coDoSet *pointSet = new coDoSet(poPoints->getObjName(), (coDistributedObject **)&pointLists.front());
    pointLists.clear();

    // set TIMESTEP attribute
    if (numTimeSteps > 1)
    {
        char buf[1024];
        sprintf(buf, "0 %i", numTimeSteps - 1);
        pointSet->addAttribute("TIMESTEP", buf);
    }

    poPoints->setCurrentObject(pointSet);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadZPR)
