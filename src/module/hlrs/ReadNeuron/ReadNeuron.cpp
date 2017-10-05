/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadNeuron module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  12.2005                                                      ++
// ++**********************************************************************/

#include <stdio.h>
#include "ReadNeuron.h"
#include <do/coDoPoints.h>

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

ReadNeuron::ReadNeuron(int argc, char **argv)
    : coSimpleModule(argc, argv, "Read IWRs Neurons")
{
    // module parameters

    m_pParamFile = addFileBrowserParam("Filename", "dummy");
    m_pParamFile->setValue("./", "*.hoc");

    m_pSomaFile = addFileBrowserParam("Somadata", "Soma File");
    m_pSomaFile->setValue("./", "*.out");

    numt = addInt32Param("numt", "Every numt timesteps");
    numt->setValue(1);

    // Output ports
    m_portLines = addOutputPort("lines", "Lines", "center lines");
    m_portLines->setInfo("Center Lines");

    m_portRad = addOutputPort("r", "Float", "Radii");
    m_portRad->setInfo("Radii");

    m_portID = addOutputPort("id", "Float", "id");
    m_portID->setInfo("ID");

    m_portSoLines = addOutputPort("SomaLines", "Lines", "soma lines");
    m_portSoLines->setInfo("Soma Lines");

    m_portSoRad = addOutputPort("SomaR", "Float", "soma radii");
    m_portSoRad->setInfo("Soma Radii");

    m_portSoData = addOutputPort("SomaData", "Float", "soma data");
    m_portSoData->setInfo("Soma Data");

    m_portSynapses = addOutputPort("synapses", "Points", "synapses");
    m_portSynapses->setInfo("Synapses");

    m_portSynID = addOutputPort("SynapseID", "Float", "synapse-id");
    m_portSynID->setInfo("SynapseID");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void ReadNeuron::getLine()
{
    if (fgets(line, LINE_SIZE, file) == NULL)
    {
        //   fprintf(stderr, "ReadNeuron::getLine(): fgets failed\n" );
    }

    c = line;
    while (*c != '\0' && isspace(*c))
    {
        c++;
    }
}

void ReadNeuron::readLine(int numSegments)
{
    int i = 0;

    char *co = oldLine + strlen(oldLine);
    while (co > oldLine && *co != '{')
        co--;
    while (co > oldLine && *co != ' ')
        co--;
    while (co > oldLine && *co == ' ')
        co--;
    co++;
    *co = '\0';
    char *name = new char[strlen(oldLine) + 1];
    strcpy(name, oldLine);

    if (strstr(name, "dendrite"))
    {
        ID = 2;
        names.push_back(name);
    }
    else if (strstr(name, "axon"))
    {
        ID = 1;
        names.push_back(name);
    }
    else if (strstr(name, "soma"))
    {
        ID = 0;
        numSomas++;
    }

    float x, y, z, r;

    if (ID != 0)
        lineList.push_back(numPoints);
    else
        somalineList.push_back(numSomaPoints);

    while (!feof(file))
    {
        getLine();
        if (strncmp(c, "pt3dadd(", 8) == 0)
        {
            int iret = sscanf(c + 8, "%f,%f,%f,%f", &x, &y, &z, &r);
            if (iret == 0)
            {
                cerr << "error scanning pt3dadd" << endl;
                break;
            }
            if (ID != 0)
            {
                xPoints.push_back(x);
                yPoints.push_back(y);
                zPoints.push_back(z);
                rad.push_back(r);
                IDs.push_back(float(ID));
                numPoints++;
            }
            else
            {
                xSomaP.push_back(x);
                ySomaP.push_back(y);
                zSomaP.push_back(z);
                somaRad.push_back(r);
                numSomaPoints++;
            }

            i++;
        }
        if (*c == '}')
        {
            if (i != numSegments + 1)
            {
                cerr << "warning wrong number of points in line " << numLines << endl;
                cerr << "is " << i << " should be " << numSegments << endl;
            }
            break;
        }
    }
    if (ID != 0)
        numLines++;
    else
        numSomaLines++;
}

void ReadNeuron::getPoint(int i, float fraction, float &x, float &y, float &z)
{
    int startIndex = 0, endIndex = 0, n;
    startIndex = lineList[i];
    if (i < numLines - 1)
    {
        endIndex = lineList[i + 1];
    }
    else
    {
        endIndex = numPoints;
    }
    float len = 0;
    float dx, dy, dz;
    for (n = startIndex + 1; n < endIndex; n++)
    {
        dx = xPoints[n - 1] - xPoints[n];
        dy = yPoints[n - 1] - yPoints[n];
        dz = zPoints[n - 1] - zPoints[n];
        len += sqrt(dx * dx + dy * dy + dz * dz);
    }
    float pos = fraction * len;
    len = 0;
    float oldLen = 0, l;
    for (n = startIndex + 1; n < endIndex; n++)
    {
        dx = xPoints[n] - xPoints[n - 1];
        dy = yPoints[n] - yPoints[n - 1];
        dz = zPoints[n] - zPoints[n - 1];
        l = sqrt(dx * dx + dy * dy + dz * dz);
        len += l;
        if (len >= pos)
        {
            float f = (pos - oldLen) / l;
            dx *= f;
            dy *= f;
            dz *= f;
            x = xPoints[n - 1] + dx;
            y = yPoints[n - 1] + dy;
            z = zPoints[n - 1] + dz;
            break;
        }
        oldLen = len;
    }
}
int ReadNeuron::compute(const char *)
{

    bool isSomaFile = true;

    // get parameters
    m_filename = new char[strlen(m_pParamFile->getValue()) + 1];
    strcpy(m_filename, m_pParamFile->getValue());

    m_somafilename = new char[strlen(m_pSomaFile->getValue()) + 1];
    strcpy(m_somafilename, m_pSomaFile->getValue());

    somafile = fopen(m_somafilename, "r");
    if (!somafile)
    {
        ///sprintf(bfr, "ERROR: can't open file %s", m_somafilename);
        //sendError(bfr);
        //return FAIL;
        isSomaFile = false;
    }

    // compute parameters
    file = fopen(m_filename, "r");
    if (!file)
    {
        sendError("ERROR: can't open file %s", m_filename);
        return FAIL;
    }

    xSomaP.clear();
    ySomaP.clear();
    zSomaP.clear();

    xPoints.clear();
    yPoints.clear();
    zPoints.clear();

    rad.clear();
    somaRad.clear();
    somaData.clear();

    IDs.clear();
    lineList.clear();
    somalineList.clear();

    names.clear();
    pX.clear();
    pY.clear();
    pZ.clear();
    synIDs.clear();

    numPoints = 0;
    numSomaPoints = 0;
    numSomas = 0;
    numLines = 0;
    numSomaLines = 0;
    numSynapses = 0;
    int numSegments;

    while (!feof(file))
    {
        if (fgets(line, LINE_SIZE, file) == NULL)
        {
            //fprintf(stderr, "ReadNeuron::compute(const char *): fgets failed\n");
        }

        char *c = line;
        while (*c != '\0' && isspace(*c))
        {
            c++;
        }
        if (strncmp(c, "nseg = ", 7) == 0)
        { // new line segment
            int iret = sscanf(c + 7, "%d", &numSegments);
            if (iret == 0)
            {
                cerr << "parse error reading nseg " << endl;
            }
            readLine(numSegments);
        }
        if (strncmp(oldLine, "objectvar Syn", 13) == 0)
        { // new synapse
            char *name = new char[strlen(line) + 1];
            strcpy(name, line);
            char *n = name;
            while (*n && *n != ' ')
                n++;
            *n = '\0';
            char *number = new char[strlen(line) + 1];
            strcpy(number, line);
            n = number;

            if (strstr(number, "AlphaSynapse"))
                synID = 0;
            else if (strstr(number, "Exp2Syn"))
                synID = 1;

            while (*n && *n != '(')
                n++;
            if (*n)
                n++;
            float fraction = 0.0;
            int ret = sscanf(n, "%f", &fraction);
            if (ret == 0)
            {
                cerr << "parse error reading fraction " << endl;
            }
            int i;
            for (i = 0; i < numLines; i++)
            {
                if (strcmp(name, names[i]) == 0)
                {
                    // we found the corresponding line
                    float x, y, z;
                    getPoint(i, fraction, x, y, z);
                    pX.push_back(x);
                    pY.push_back(y);
                    pZ.push_back(z);

                    synIDs.push_back(float(synID));

                    numSynapses++;
                    break;
                }
            }
            delete[] name;
            delete[] number;
        }
        strcpy(oldLine, line);
    }

    int whichlines = numt->getValue();
    timesteps = 0;
    float sdata;
    int k, numline = 0, getline = whichlines;

#ifndef YAC
    char name[256];
#endif

    if (isSomaFile)
    {
        while (!feof(somafile))
        {

            if (numline != 0)
            {
                while ((numline < (getline - 1)) && (!feof(somafile)))
                {
                    if (fgets(line, SO_LINE_SIZE, somafile) == NULL)
                    {
                        //
                    }
                    else
                        numline++;
                }
                getline += whichlines;
            }

            if (fgets(line, SO_LINE_SIZE, somafile) == NULL)
            {
                //
            }
            else
            {
                numline++;
                char *ch = line;
                while (*ch != '\0' && isspace(*ch))
                {
                    ch++;
                }
                if (sscanf(ch, "%f", &sdata))
                {
                    while (!isspace(*ch))
                    {
                        ch++;
                    }
                }
                for (k = 0; k < numSomas; k++)
                {
                    while (*ch != '\0' && isspace(*ch))
                    {
                        ch++;
                    }
                    if (sscanf(ch, "%f", &sdata))
                    {
                        somaData.push_back(sdata);
                        while (!isspace(*ch))
                        {
                            ch++;
                        }
                    }
                }
#ifndef YAC
                sprintf(name, "%s_%5d", m_portSoData->getObjName(), timesteps);
#else
                coObjInfo name = m_portSoData->getNewObjectInfo();
#endif
                coDoFloat *data = new coDoFloat(name, numSomas, &somaData[0]);
                somaDataList.push_back(data);
                somaData.clear();
                timesteps++;
            }
        }
    }
    else
    {
        for (k = 0; k < numSomas; k++)
        {
            somaData.push_back(1);
        }
    }

    // construct the output data
    int *vl = new int[numPoints];
    int i;
    for (i = 0; i < numPoints; i++)
        vl[i] = i;

    int *vls = new int[numSomaPoints];
    int j;
    for (j = 0; j < numSomaPoints; j++)
        vls[j] = j;

    if (numPoints)
    {
        coDoLines *linesObj = new coDoLines(m_portLines->getObjName(), numPoints, &xPoints[0], &yPoints[0], &zPoints[0], numPoints, vl, numLines, &lineList[0]);

        coDoFloat *dataObj = new coDoFloat(m_portRad->getObjName(), numPoints, &rad[0]);

        coDoFloat *idDataObj = new coDoFloat(m_portID->getObjName(), numPoints, &IDs[0]);
        delete[] vl;

        // Assign sets to output ports:
        m_portLines->setCurrentObject(linesObj);
        m_portRad->setCurrentObject(dataObj);
        m_portID->setCurrentObject(idDataObj);
    }
    if (numSomaPoints)
    {

        if (isSomaFile)
        {
#ifndef YAC
            char name[256];
            snprintf(name, 256, "%s_%5d", m_portSoLines->getObjName(), timesteps);
#else
            coObjInfo name = m_portSoLines->getNewObjectInfo();
            name.timeStep = -1;
            name.numTimeSteps = 0;
#endif
            coDoLines *somaLinesObj = new coDoLines(name, numSomaPoints, &xSomaP[0], &ySomaP[0], &zSomaP[0], numSomaPoints, vls, numSomaLines, &somalineList[0]);

#ifndef YAC
            snprintf(name, 256, "%s_%5d", m_portSoRad->getObjName(), timesteps);
#else
            name = m_portSoRad->getNewObjectInfo();
            name.timeStep = -1;
            name.numTimeSteps = 0;
#endif
            coDoFloat *somaRadObj = new coDoFloat(name, numSomaPoints, &somaRad[0]);

            coDistributedObject **somaLineObjects = new coDistributedObject *[timesteps + 1];
            for (i = 0; i < timesteps; i++)
            {
                somaLineObjects[i] = somaLinesObj;
#ifndef YAC
                if (i > 0)
                    somaLinesObj->incRefCount();
#endif
            }
            somaLineObjects[timesteps] = NULL;
            coDistributedObject **somaRadObjects = new coDistributedObject *[timesteps + 1];
            for (i = 0; i < timesteps; i++)
            {
#ifndef YAC
                if (i > 0)
                    somaRadObj->incRefCount();
#endif
                somaRadObjects[i] = somaRadObj;
            }
            somaRadObjects[timesteps] = NULL;
            coDistributedObject **somaDataObjects = new coDistributedObject *[timesteps + 1];

            somaData_Iter = somaDataList.begin();
            for (i = 0; i < timesteps; i++)
            {
                coDoFloat *o = *somaData_Iter;
                somaDataObjects[i] = o;
#ifdef YAC

                o->getHdr()->setBlock(0, 1);
                o->getHdr()->setTime(i, timesteps);
                o->getHdr()->setRealTime((float)i);
                float *data;
                o->getAddress(&data);
#endif
                somaData_Iter++;
            }
            somaDataObjects[timesteps] = NULL;

            coDoSet *lineSet = new coDoSet(m_portSoLines->getNewObjectInfo(), somaLineObjects);
            coDoSet *radSet = new coDoSet(m_portSoRad->getNewObjectInfo(), somaRadObjects);
            coDoSet *dataSet = new coDoSet(m_portSoData->getNewObjectInfo(), somaDataObjects);

            lineSet->addAttribute("TIMESTEP", "1 4002");
            radSet->addAttribute("TIMESTEP", "1 4002");
            dataSet->addAttribute("TIMESTEP", "1 4002");

            // Assign sets to output ports:
            m_portSoLines->setCurrentObject(lineSet);
            m_portSoRad->setCurrentObject(radSet);
            m_portSoData->setCurrentObject(dataSet);
        }
        else
        {

            coDoLines *somaLinesObj = new coDoLines(m_portSoLines->getObjName(), numSomaPoints, &xSomaP[0], &ySomaP[0], &zSomaP[0], numSomaPoints, vls, numSomaLines, &somalineList[0]);
            coDoFloat *somaRadObj = new coDoFloat(m_portSoRad->getObjName(), numSomaPoints, &somaRad[0]);
            coDoFloat *somaDataObj = new coDoFloat(m_portSoData->getObjName(), numSomas, &somaData[0]);

            // Assign sets to output ports:
            m_portSoLines->setCurrentObject(somaLinesObj);
            m_portSoRad->setCurrentObject(somaRadObj);
            m_portSoData->setCurrentObject(somaDataObj);
        }
        delete[] vls;
    }
    if (numSynapses)
    {
        coDoPoints *synDataObj = new coDoPoints(m_portSynapses->getObjName(), numSynapses, &pX[0], &pY[0], &pZ[0]);

        coDoFloat *synIdDataObj = new coDoFloat(m_portSynID->getObjName(), numSynapses, &synIDs[0]);

        m_portSynapses->setCurrentObject(synDataObj);
        m_portSynID->setCurrentObject(synIdDataObj);
    }

    if (isSomaFile)
        fclose(somafile);

    fclose(file);

    somaDataList.clear();

    for (i = 0; i < numLines; i++)
        delete[] names[i];

    return SUCCESS;
}

MODULE_MAIN(Reader, ReadNeuron)
