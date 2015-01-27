/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE ReadAramcoSim
//
// Initial version: 2001-10-26 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ext2SV.h"
#include "ReadAramcoSim.h"
#include "AramcoSimFile.h"
#include <string.h>
#include <errno.h>
#include <fstream.h>
#include <stdlib.h>

// dump field with specified size to filename
static void dumpField(const float *field, const char *filename,
                      int numCol, int numRow, int numLay,
                      int numSpec)
{
    int lay, row, col, spec;
    ofstream actStream(filename);
    for (lay = 0; lay < numLay; lay++)
    {
        for (row = 0; row < numRow; row++)
        {
            for (col = 0; col < numCol; col++)
            {
                actStream << "col=" << col
                          << " row=" << row
                          << " lay=" << lay
                          << " :  ";
                for (spec = 0; spec < numSpec; spec++)
                {
                    actStream << " " << *field;
                    field++;
                }
                actStream << endl;
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadAramcoSim::ReadAramcoSim()
    : coModule("ARAMCO Reader")
{
    // File name
    p_filename = addFileBrowserParam("filename", "Input File");

    //p_filename->setValue("data/aramco/testCOVISE.sim","*.sim");
    p_filename->setImmediate(1);

    // Fixed output species
    p_mesh = addOutputPort("Mesh", "coDoUnstructuredGrid", "Grid");
    p_meshTime = addOutputPort("MeshTime", "coDoUnstructuredGrid", "Timestepped Grid");
    p_strMesh = addOutputPort("StrMesh", "coDoStructuredGrid", "Structured Grid");

    // Scaling of z coordinates
    p_zScale = addFloatParam("zScale", "z scaling factor");
    p_zScale->setValue(5.0);

    // Loop for data fields: choices and ports
    int i;
    char name[32];
    const char *defaultChoice[] = { "---" };
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(name, "Choice_%d", i);
        p_choice[i] = addChoiceParam(name, "Select data for port");
        p_choice[i]->setValue(1, defaultChoice, 1);
        p_choice[i]->setImmediate(1);

        sprintf(name, "Data_%d", i);
        p_data[i] = addOutputPort(name, "coDoFloat", name);
    }

    // initialize
    d_simFile = NULL;
}

void ReadAramcoSim::postInst()
{
    p_zScale->show();
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  param() is called once for every immediate parameter change
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadAramcoSim::param(const char *paraname)
{
    // read new file
    if (0 == strcmp(paraname, p_filename->getName())
        && !in_map_loading())
    {

        // read the file: creates own error message, ignore result here
        readFile();
    }
}

int ReadAramcoSim::readFile()
{
    // delete old file (if there)
    delete d_simFile;

    // get new one
    d_simFile = new AramcoSimFile(p_filename->getValue());

    // check for errors
    if (!d_simFile)
    {
        sendError("Unknown error creating SIM file object");
        return -1;
    }
    if (d_simFile->isBad())
    {
        sendError(d_simFile->getErrorMessage());
        delete d_simFile;
        d_simFile = NULL;
        return -1;
    }

    // get choice labels, keep value if legal
    int i;
    int numChoice = d_simFile->numDataSets() + 1;
    const char *const *labels = d_simFile->getLabels();
    for (i = 0; i < NUM_PORTS; i++)
    {
        int oldVal = p_choice[i]->getValue();
        if (oldVal <= numChoice)
            p_choice[i]->setValue(numChoice, labels, oldVal);
        else
            p_choice[i]->setValue(numChoice, labels, 1);
        p_choice[i]->show();
    }

    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadAramcoSim::compute()
{
    int row, col, lay;

    // no file read yet ?
    if (!d_simFile)
    {

        // try to read
        if (readFile() != 0)
        {
            return STOP_PIPELINE;
        }
    }

    /////////////////////// MESH ///////////////////////

    // get the sizes
    int numRow, numCol, numLay;
    d_simFile->getSize(numLay, numRow, numCol);

    // get the Coordinates
    const float *xyCoord = d_simFile->getXYcoord();
    const float *zCoord = d_simFile->getZcoord();

    // debug output
    const char *debugSetting = getenv("ARAMCO_READER_DEBUG");
    if (debugSetting)
    {
        if (strstr(debugSetting, "xy"))
        {
            dumpField(xyCoord, "DEBUG-xy", numCol, numRow, 1, 2);
        }
        if (strstr(debugSetting, "z"))
        {
            dumpField(zCoord, "DEBUG-z", numCol, numRow, numLay, 1);
        }
    }

    // get the activation info
    const int *activeMap = d_simFile->getActiveMap();
    int numActive = d_simFile->numActive();

    // retrieve numvber of time steps
    int numSteps = d_simFile->numTimeSteps();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // create structured grid
    coDoStructuredGrid *strGrid
        = new coDoStructuredGrid(p_strMesh->getObjName(), numLay, numRow, numCol);

    float *strX, *strY, *strZ;
    strGrid->getAddresses(&strX, &strY, &strZ);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // create unstructured grid
    coDoUnstructuredGrid *usg
        = new coDoUnstructuredGrid(p_mesh->getObjName(), numActive, 8 * numActive,
                                   numLay * numRow * numCol, 1);

    int *usgConn, *usgElem, *usgType;
    float *usgX, *usgY, *usgZ;

    usg->getAddresses(&usgElem, &usgConn, &usgX, &usgY, &usgZ);
    usg->getTypeList(&usgType);
    // glob=global Field, loc=field without de-activated Cells
    int globIdx = 0, locIdx = 0;
    for (lay = 0; lay < numLay - 1; lay++)
    {
        for (row = 0; row < numRow - 1; row++)
        {
            for (col = 0; col < numCol - 1; col++)
            {
                if (activeMap[globIdx] >= 0)
                {
                    int base = col + (numCol * row) + (numCol * numRow * lay);

                    // Element table: All HEX elems
                    *usgElem = 8 * locIdx;
                    usgElem++;
                    *usgType = TYPE_HEXAEDER;
                    usgType++;

                    // Vertex table
                    *usgConn = base;
                    usgConn++;
                    *usgConn = base + 1;
                    usgConn++;
                    *usgConn = base + numCol + 1;
                    usgConn++;
                    *usgConn = base + numCol;
                    usgConn++;
                    base += numCol * numRow;
                    *usgConn = base;
                    usgConn++;
                    *usgConn = base + 1;
                    usgConn++;
                    *usgConn = base + numCol + 1;
                    usgConn++;
                    *usgConn = base + numCol;
                    usgConn++;

                    // one REAL cell more
                    locIdx++;
                }

                // one global cell more
                globIdx++;
            }
        }
    }

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // create all Vertices : x/y coords from xyCoord, z coords from zCoord
    int xyIdx;
    globIdx = 0;
    float zScaleFact = p_zScale->getValue();

    // loop over all Layers
    for (lay = 0; lay < numLay; lay++)
    {

        // X/Y coords are the same on all layers
        xyIdx = 0;
        for (row = 0; row < numRow; row++)
        {
            for (col = 0; col < numCol; col++)
            {
                usgX[globIdx] = xyCoord[xyIdx];
                usgY[globIdx] = xyCoord[xyIdx + 1];
                usgZ[globIdx] = zCoord[globIdx] * zScaleFact;

                strX[globIdx] = xyCoord[xyIdx];
                strY[globIdx] = xyCoord[xyIdx + 1];
                strZ[globIdx] = zCoord[globIdx] * zScaleFact;

                xyIdx += 2;
                globIdx++;
            }
        }
    }

    p_mesh->setCurrentObject(usg);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Time-dependent mesh: set with multiple time the same grid
    coDistributedObject **doArr = new coDistributedObject *[numSteps + 1];
    int step;
    for (step = 0; step < numSteps; step++)
    {
        doArr[step] = usg;
        usg->incRefCount();
    }
    doArr[numSteps] = NULL;
    coDoSet *set = new coDoSet(p_meshTime->getObjName(), doArr);
    char attribVal[16];
    sprintf(attribVal, "0 %d", numSteps - 1);
    set->addAttribute("TIMESTEP", attribVal);

    p_meshTime->setCurrentObject(set);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Time-dependent data: set with multiple time the same grid

    int port;
    for (port = 0; port < NUM_PORTS; port++)
    {

        // -1 for counting from 1,
        int fieldNo = p_choice[port]->getValue() - 2;
        // -1 for "---" setting

        // no output to create - next port
        if (fieldNo == -1)
        {
            // Connected, but no output: warn the guys
            if (p_data[port]->isConnected())
            {
                sendWarning("No data field selected for connected port '%s'",
                            p_data[port]->getName());
            }
            continue;
        }

        // transient data
        if (d_simFile->isTransient(fieldNo))
        {
            char transObjName[1024];
            const char *objName = p_data[port]->getObjName();
            for (step = 0; step < numSteps; step++)
            {
                sprintf(transObjName, "%s_%d", objName, step);
                doArr[step] = readField(transObjName, fieldNo, step);
            }
            doArr[numSteps] = NULL;
            coDoSet *set = new coDoSet(objName, doArr);
            char attribVal[16];
            sprintf(attribVal, "0 %d", numSteps - 1);
            set->addAttribute("TIMESTEP", attribVal);
            p_data[port]->setCurrentObject(set);
        }

        // stationary data
        else
        {
            p_data[port]->setCurrentObject(readField(p_data[port]->getObjName(), fieldNo, 0));
        }

        p_data[port]->getCurrentObject()->addAttribute("SPECIES", p_choice[port]->getActLabel());

    } // loop over ports

    delete[] doArr;

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Read a single data field into named object
coDoFloat *
ReadAramcoSim::readField(const char *name, int fieldNo, int stepNo)
{
    // get the sizes
    int numRow, numCol, numLay;
    d_simFile->getSize(numLay, numRow, numCol);

    int numElem = (d_simFile->isCellBased(fieldNo))
                      ? (numRow - 1) * (numCol - 1) * (numLay - 1)
                      : numRow * numCol * numLay;

    coDoFloat *dataObj
        = new coDoFloat(name, numElem);

    float *doDataSpace;
    dataObj->getAddress(&doDataSpace);
    d_simFile->readData(doDataSpace, fieldNo, stepNo);

    return dataObj;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    ReadAramcoSim *application = new ReadAramcoSim;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
