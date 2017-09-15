/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadIFF module                                         ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  10.2005                                                      ++
// ++**********************************************************************/

#include <stdio.h>
#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include "ReadIFF.h"
#include <vector>

#include <float.h>
#include <limits.h>
#include <string.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ReadIFF::ReadIFF(int argc, char *argv[])
    : coModule(argc, argv, "Read ITT's simulation data")
{
    // module parameters

    m_pParamFile = addFileBrowserParam("Filename", "dummy");

    m_pParamFile->setValue("./", "*.txt");

    // Output ports
    m_portPoints = addOutputPort("points", "Points", "points Output");
    m_portPoints->setInfo("points Output");

    m_portColors = addOutputPort("colors", "RGBA", "Atom Colors Output");
    m_portColors->setInfo("Colors Output");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadIFF::compute(const char *)
{
    char line[LINE_SIZE];
    //float positions[MAXTIMESTEPS][3];
    //int tmpColors[MAXTIMESTEPS];

    std::vector<int *> tsColors;
    std::vector<float *> tsPoints;

    // get parameters
    m_filename = new char[strlen(m_pParamFile->getValue()) + 1];
    strcpy(m_filename, m_pParamFile->getValue());

    // compute parameters
    FILE *file = Covise::fopen(m_filename, "r");
    if (!file)
    {
        Covise::sendError("ERROR: can't open file %s", m_filename);
        return FAIL;
    }
    float oc[3];
    float ac[3];
    int col;
    int i, j;
    float *xCoord;
    float *yCoord;
    float *zCoord;
    int *atomColors;
    std::vector<int> atomCounts;
    std::vector<int *> colors;
    std::vector<float *> xPoints;
    std::vector<float *> yPoints;
    std::vector<float *> zPoints;

    int numAtoms = 0;
    int numTimeSteps = 0;
    while (!feof(file))
    {
        char *unread = fgets(line, LINE_SIZE, file);
        cerr << unread << endl;
        int iret = sscanf(line, "%f,%f,%f,%f,%f,%f,%i", &oc[0], &oc[1], &oc[2], &ac[0], &ac[1], &ac[2], &col);
        if (iret == 0)
            break;

        xCoord = new float[numAtoms + 1];
        yCoord = new float[numAtoms + 1];
        zCoord = new float[numAtoms + 1];
        atomColors = new int[numAtoms + 1];
        xPoints.push_back(xCoord);
        yPoints.push_back(yCoord);
        zPoints.push_back(zCoord);
        colors.push_back(atomColors);
        if (numTimeSteps > 0)
        {
            // copy atoms from last timestep
            memcpy(xCoord, xPoints[numTimeSteps - 1], numAtoms * sizeof(float));
            memcpy(yCoord, yPoints[numTimeSteps - 1], numAtoms * sizeof(float));
            memcpy(zCoord, zPoints[numTimeSteps - 1], numAtoms * sizeof(float));
            memcpy(atomColors, colors[numTimeSteps - 1], numAtoms * sizeof(float));
        }
        if (oc[0] == 0 && oc[1] == 0 && oc[2] == 0)
        {
            // add new atom
            xCoord[numAtoms] = ac[0];
            yCoord[numAtoms] = ac[1];
            zCoord[numAtoms] = ac[2];
            numAtoms++;
        }
        else
        {
            // find old atom and replace
            for (i = 0; i < numAtoms; i++)
            {
                if (xCoord[i] == oc[0] && yCoord[i] == oc[1] && zCoord[i] == oc[2])
                {
                    xCoord[i] = ac[0];
                    yCoord[i] = ac[1];
                    zCoord[i] = ac[2];
                    break;
                }
            }
        }
        atomCounts.push_back(numAtoms);
        numTimeSteps++;
    }

    // construct the output data

    const char *obj_points;
    const char *obj_atomcolors;

    // name handles
    obj_points = m_portPoints->getObjName();
    obj_atomcolors = m_portColors->getObjName();

    coDoPoints **pointsObj = new coDoPoints *[numTimeSteps + 1];
    coDoRGBA **colorObj = new coDoRGBA *[numTimeSteps + 1];
    pointsObj[numTimeSteps] = 0;
    colorObj[numTimeSteps] = 0;
    for (j = 0; j < numTimeSteps; j++)
    {
        char buf[512];
        snprintf(buf, sizeof(buf), "%stimeStep%d", obj_points, j);

        pointsObj[j] = new coDoPoints(buf, atomCounts[j], xPoints[j], yPoints[j], zPoints[j]);
        snprintf(buf, sizeof(buf), "%stimeStep%d", obj_atomcolors, j);

        colorObj[j] = new coDoRGBA(buf, atomCounts[j], colors[j]);
        delete[] xPoints[j];
        delete[] yPoints[j];
        delete[] zPoints[j];
        delete[] colors[j];
    }

    char cbuf[1024];
    snprintf(cbuf, sizeof(cbuf), "0 %d", numberOfTimesteps - 1);

    // Create set objects:
    coDoSet *setPoints = NULL;
    coDoSet *setColors = NULL;

    setPoints = new coDoSet(m_portPoints->getObjName(), (coDistributedObject **)pointsObj);
    setColors = new coDoSet(m_portColors->getObjName(), (coDistributedObject **)colorObj);

    // Set timestep attribute:
    setPoints->addAttribute("TIMESTEP", cbuf);
    setColors->addAttribute("TIMESTEP", cbuf);

    // Assign sets to output ports:
    m_portPoints->setCurrentObject(setPoints);
    m_portColors->setCurrentObject(setColors);
    fclose(file);

    return SUCCESS;
}

MODULE_MAIN(IO, ReadIFF)
