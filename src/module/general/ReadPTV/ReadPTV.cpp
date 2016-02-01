/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read PTV data from DLR.                      **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                          Uwe Woessner                                  **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Nobelstrasse 19                                **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 01.02.2016                                               **
\**************************************************************************/

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <api/coModule.h>
#include <util/coFileUtil.h>
#include <alg/coChemicalElement.h>
#include <limits.h>
#include <float.h>
#include <cassert>
#include "ReadPTV.h"

/// Constructor
coReadPTV::coReadPTV(int argc, char *argv[])
    : coModule(argc, argv, "Read PTV files containing lists of particle positions and their velocities.")
{

    // Create ports:
    poPoints = addOutputPort("Location", "Points", "particle positions");
    poTypes = addOutputPort("Number", "Int", "particle number");
    poVelos = addOutputPort("Velocities", "Vec3", "Particle velocities");

    // Create parameters:
    pbrFilename = addFileBrowserParam("Filename", "PTV file");
    pbrFilename->setValue("data/", "*.ptv;*.PTV/*.*");

    pLimitTimesteps = addInt32Param("LimitTimestep", "Maximum number of timesteps to read (0 = all)");
    pLimitTimesteps->setValue(0);

}

/// Compute routine: load checkpoint file
int coReadPTV::compute(const char *)
{



    
    int timestepLimit = pLimitTimesteps->getValue();
    const char *path = pbrFilename->getValue();
    const char *fileName = coDirectory::fileOf(path);
    const char *dirName = coDirectory::dirOf(path);
    int fileNumber=-1;
    sscanf(fileName,"%d-",&fileNumber);
    if(fileNumber < 0)
    {
        sendError("Filename does not start with a number: %s",fileName);
        return STOP_PIPELINE;
    }
    int numFiles =0;
        
    coDirectory *dir= coDirectory::open(dirName);
    if(dir == NULL)
    {
        sendError("could not open directory: %s",dirName);
        return STOP_PIPELINE;
    }
    
    std::vector<coDoPoints *> points;
    std::vector<coDoVec3 *> velos;
    std::vector<coDoInt *> types;

    while(timestepLimit == 0 || numFiles < timestepLimit)
    {
        FILE *fp=NULL;
        int numSkipped=0;
        do {
            for(int i=0;i<dir->count();i++)
            {
                char *nextFileStart = new char[strlen(fileName)+5];
                sprintf(nextFileStart,"%d-",fileNumber);
                int len = strlen(nextFileStart);
                if(strncmp(dir->name(i),nextFileStart,len)==0)
                {
                    char *fullPath = new char[strlen(dir->name(i))+strlen(dirName)+5];
                    sprintf(fullPath,"%s/%s",dirName,dir->name(i));
                    fp=fopen(fullPath,"r");
                    break;
                }
            }
            if(fp == NULL)
            {
                numSkipped++;
                fileNumber++;
            }
        } while(fp==NULL && numSkipped < 20);
        if(fp==NULL)
            break;
        // read one timestep
        std::vector<int> pNumber;
        std::vector<float> px,py,pz, vx, vy, vz;
        while (!feof(fp))
        {
            char buf[1024];
            float x, y, z, u,v,w;
            int num;
            fgets(buf,1024,fp);
            int n = sscanf(buf, "%d %f %f %f %f %f %f\n", &num, &x, &y, &z, &u, &v, &w);
            if (n != 7)
            {
                break;
            }
            
            px.push_back(x);
            py.push_back(y);
            pz.push_back(z);
            vx.push_back(u);
            vy.push_back(v);
            vz.push_back(w);
            pNumber.push_back(num);
        }
        // done reading the file
        fclose(fp);
        
        char name[1024];
        snprintf(name, sizeof(name), "%s_%d", poTypes->getObjName(), fileNumber);
        coDoInt *doTypes = new coDoInt(name, pNumber.size(), &pNumber[0]);

        snprintf(name, sizeof(name), "%s_%d", poPoints->getObjName(), fileNumber);
        coDoPoints *doPoints = new coDoPoints(name, px.size(), &px[0], &py[0], &pz[0]);
        
        snprintf(name, sizeof(name), "%s_%d", poVelos->getObjName(), fileNumber);
        coDoVec3 *doVelos = new coDoVec3(name, vx.size(), &vx[0], &vy[0], &vz[0]);

        
        points.push_back(doPoints);
        velos.push_back(doVelos);
        types.push_back(doTypes);
        // done creating data objects
        fileNumber++;
        numFiles++;
    }

    
    // Create set objects:
    coDoSet *setPoints = new coDoSet(poPoints->getObjName(), points.size(), (coDistributedObject **)&points[0]);
    coDoSet *setVelos = new coDoSet(poVelos->getObjName(), velos.size(), (coDistributedObject **)&velos[0]);
    coDoSet *setTypes = new coDoSet(poTypes->getObjName(), types.size(), (coDistributedObject **)&types[0]);
    // Now the arrays can be cleared:
    points.clear();
    velos.clear();
    types.clear();

    // Set timestep attribute:
    if (numFiles > 1)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%d %d", 0, numFiles - 1);
        setPoints->addAttribute("TIMESTEP", buf);
        setVelos->addAttribute("TIMESTEP", buf);
        setTypes->addAttribute("TIMESTEP", buf);
    }

    // Assign sets to output ports:
    poPoints->setCurrentObject(setPoints);
    poVelos->setCurrentObject(setVelos);
    poTypes->setCurrentObject(setTypes);
    
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadPTV)
