/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)2001     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Björn Sander/Uwe                                               **
 ** =============================================================================
 ** READRFL Modul zum Lesen von ANSYS RFL-Ergebnisfiles (FLOWTRAN)
 ** -----------------------------------------------------------------------------
 ** 17.9.2001  Björn Sander
 ** =============================================================================
 **                           **
 **                                                                        **
\**************************************************************************/
#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <util/coviseCompat.h>
#include "ReadRFL.h"
#include "ReadRflFile.h"

extern const char *dofname[32];

extern const char *exdofname[28];

ReadRFL::ReadRFL(int argc, char *argv[])
    : coModule(argc, argv, "Reads stationary and instationary ANSYS Flotran Result files")
{

    // the output port
    gridPort = addOutputPort("unsgrid", "UnstructuredGrid", "unstructured Grid");

    // select the OBJ file name with a file browser
    rflFileParam = addFileBrowserParam("rflFile", "Flotran result file");
    rflFileParam->setValue("data/hlrs/anna/test.rfl", "*.rfl");

    datasetNum = addIntSliderParam("datasetNum", "Dataset number");
    datasetNum->setValue(1, 2, 2);
    numTimesteps = addIntSliderParam("numTimesteps", "Number of timesteps to read");
    numTimesteps->setValue(1, 1, 1);
    numSkip = addInt32Param("numSkip", "Number of timesteps to skip");
    numSkip->setValue(0);
    typeList = NULL;
    rflFile = NULL;
    int i;
    static const char *defFieldVal[] = { "---" };
    char buffer[500];
    for (i = 0; i < NUMPORTS; i++)
    {
        sprintf(buffer, "dof_%d", i);
        dofs[i] = addChoiceParam(buffer, "DOF to read for output");
        dofs[i]->setValue(1, defFieldVal, 0);

        sprintf(buffer, "data_%d", i);
        data[i] = addOutputPort(buffer,
                                "Float|Vec3", "Data Output");
    }
}

ReadRFL::~ReadRFL()
{
}

void ReadRFL::quit()
{
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            I M M E D I A T E   C A L L B A C K
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

void ReadRFL::param(const char *paramname, bool /*inMapLoading*/)
{

    char realFile[1000];
    const char *filename;
    if (strcmp(rflFileParam->getName(), paramname) == 0)
    {
        filename = rflFileParam->getValue();
        if (filename != NULL)
        {
            fprintf(stderr, "Reading file: %s\n", filename);
            Covise::getname(realFile, filename);
            if (rflFile)
            {
                fclose(rflFile->rfp);
                rflFile->rfp = NULL;
                delete rflFile;
            }
            rflFile = new READRFL;

            switch (rflFile->OpenFile(realFile))
            {
            // 0    : alles OK
            // 1    : File not found
            // 2    : could not read header
            // 3    : Read Error Nodal equivalence
            // 4    : Read Error Element equivalence
            // 5    : Read Error Time table
            case 0:
                sendInfo("Open file OK\n");
                break;

            case 1:
                sendError("Open file: file not found \n");
                return;

            case 2:
                sendError("Open file: Read Error, header \n");
                return;

            case 3:
                sendError("Open file: Read Error, nodal equ tabular \n");
                return;

            case 4:
                sendError("Open file: Read Error, element equi tabular \n");
                return;

            case 5:
                sendError("Open file: Read Error, time table \n");
                return;
            }
            if (rflFile->rstheader.numsets > 1)
                datasetNum->setValue(1, rflFile->rstheader.numsets, 2);
            else
                datasetNum->setValue(1, rflFile->rstheader.numsets, rflFile->rstheader.numsets);

            updateChoices();
        }
        else
        {
            sendError("ERROR: fileName is NULL");
        }
    }
    else if (strcmp(datasetNum->getName(), paramname) == 0)
    {
        updateChoices();
    }
}

void ReadRFL::updateChoices()
{

    int i;
    switch (rflFile->ReadSHDR(datasetNum->getValue()))
    {
    case 0:
        sendInfo("ReadSHDR OK\n");
        break;
    default:
        sendError("ReadSHDR: Error\n");
        break;
    }
    int numChoices = rflFile->solheader.numdofs + rflFile->solheader.numexdofs + 2;
    char **choiceLabels = new char *[numChoices];
    choiceLabels[0] = new char[5];
    strcpy(choiceLabels[0], "None");
    choiceLabels[1] = new char[9];
    strcpy(choiceLabels[1], "Velocity");
    for (i = 2; i < numChoices; i++)
    {
        if ((i - 2) < rflFile->solheader.numdofs)
        {
            choiceLabels[i] = new char[strlen(dofname[rflFile->solheader.dof[i - 2] - 1]) + 1];
            strcpy(choiceLabels[i], dofname[rflFile->solheader.dof[i - 2] - 1]);
        }
        else
        {
            choiceLabels[i] = new char[strlen(exdofname[rflFile->solheader.exdof[i - 2 - rflFile->solheader.numdofs] - 1]) + 1];
            strcpy(choiceLabels[i], exdofname[rflFile->solheader.exdof[i - 2 - rflFile->solheader.numdofs] - 1]);
        }
    }
    for (i = 0; i < NUMPORTS; i++)
    {
        int currentChoice = dofs[i]->getValue();
        if (currentChoice >= numChoices)
        {
            dofs[i]->setValue(numChoices, choiceLabels, 0);
        }
        else
        {
            dofs[i]->setValue(numChoices, choiceLabels, currentChoice);
        }
    }

    for (i = 0; i < numChoices; i++)
    {
        delete[] choiceLabels[i];
    }
    delete[] choiceLabels;
}

int ReadRFL::compute(const char *)
{
    int i, j;
    int numTimestepsRead = 0;
    int timestep = datasetNum->getValue();
    coDistributedObject **grids = new coDistributedObject *[numTimesteps->getValue() + 1];
    coDistributedObject ***dataObjects = new coDistributedObject **[NUMPORTS];
    for (i = 0; i < NUMPORTS; i++)
    {
        dataObjects[i] = new coDistributedObject *[numTimesteps->getValue() + 1];
        memset(dataObjects[i], 0, (numTimesteps->getValue() + 1) * sizeof(coDistributedObject *));
    }
    memset(grids, 0, (numTimesteps->getValue() + 1) * sizeof(coDistributedObject *));

    while (numTimestepsRead < numTimesteps->getValue())
    {
        int result = rflFile->GetDataset(timestep);
        switch (result)
        {
        // 1        : File ist nicht offen/initialisiert
        // 2        : Read Error DSI-Tabelle
        // 3        : Num ist nicht im Datensatz
        // 4        : Read Error Solution Header
        // 5        : Read Error DOFs
        // 6        : Read Error exDOFs
        case 0:
            sendInfo("GetData : OK\n");
            break;

        case 1:
            sendInfo("GetData : file not open\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 2:
            sendInfo("GetData : read error: DSI\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 3:
            sendInfo("GetData : num exeeds limits\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 4:
            sendInfo("GetData : read error solution header\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 5:
            sendInfo("GetData : read error DOFs\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 6:
            sendInfo("GetData : read error exDOF\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;
        }
        if (result != 0)
        {
            break;
        }
        result = rflFile->GetNodes();
        switch (result)
        {
        // 1        : Read Error Geometrieheader
        // 2        : Read Error Nodes
        // 3        : Read Error Elementbeschreibung
        // 4        : Read Error ETYs
        // 5        : Read Error Elementtabelle
        // 6        : Read Error Elemente
        case 0:
            sendInfo("GetNodes : ok\n");
            break;

        case 1:
            sendInfo("GetNodes : read error geo\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 2:
            sendInfo("GetNodes : read error nodes\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 3:
            sendInfo("GetNodes : read error element description\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 4:
            sendInfo("GetNodes : read error ety\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 5:
            sendInfo("GetNodes : read error element tabular\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;

        case 6:
            sendInfo("GetNodes : read error elements\n");
            if (numTimesteps->getValue() > 1)
                break;
            else
                return FAIL;
        }
        if (result != 0)
        {
            break;
        }

        typeList = new int[rflFile->anzelem];
        numVertices = 0;
        for (i = 0; i < rflFile->anzelem; ++i)
        {
            switch (rflFile->element[i].anznodes)
            {
            case 4: // 2D
                if (rflFile->element[i].nodes[3] == rflFile->element[i].nodes[2])
                {

                    typeList[i] = TYPE_TRIANGLE;
                    rflFile->element[i].anznodes = 3;
                    // Nodes selber ist ok, die hinteren beiden sind ja doppelt
                }
                else
                {
                    // viereck
                    typeList[i] = TYPE_QUAD;
                }
                break;

            case 8: // 3d Fluid
                if (rflFile->element[i].nodes[4] != rflFile->element[i].nodes[5])
                {
                    if (rflFile->element[i].nodes[6] == rflFile->element[i].nodes[7])
                    {
                        typeList[i] = TYPE_PRISM;
                        rflFile->element[i].anznodes = 6;
                        // lösche knoten 3, rest eins vor
                        rflFile->element[i].nodes[3] = rflFile->element[i].nodes[4];
                        rflFile->element[i].nodes[4] = rflFile->element[i].nodes[5];
                        rflFile->element[i].nodes[5] = rflFile->element[i].nodes[6];
                    }
                    else
                    {
                        typeList[i] = TYPE_HEXAEDER;
                        // alles ok
                    }
                }
                else
                {
                    if (rflFile->element[i].nodes[2] == rflFile->element[i].nodes[3])
                    {
                        typeList[i] = TYPE_TETRAHEDER;
                        rflFile->element[i].anznodes = 4;
                        rflFile->element[i].nodes[3] = rflFile->element[i].nodes[4];
                    }
                    else
                    {
                        typeList[i] = TYPE_PYRAMID;
                        rflFile->element[i].anznodes = 5;
                        // Knoten sind OK
                    }
                }
                break;

            default: // FIXME: macht Fehler bei Mittelknoten
                break;
            }
            numVertices += rflFile->element[i].anznodes;
        }

        if (numTimesteps->getValue() > 1)
        {
            char *objectName = new char[strlen(gridPort->getObjName()) + 50];
            sprintf(objectName, "%s_%d", gridPort->getObjName(), timestep);
            gridObject = new coDoUnstructuredGrid(objectName, rflFile->anzelem, numVertices, rflFile->anznodes, true);
            grids[numTimestepsRead] = gridObject;
        }
        else
        {
            gridObject = new coDoUnstructuredGrid(gridPort->getObjName(), rflFile->anzelem, numVertices, rflFile->anznodes, true);
        }

        gridObject->getAddresses(&el, &vl, &x_c, &y_c, &z_c);
        gridObject->getTypeList(&tl);

        numVertices = 0;
        for (i = 0; i < rflFile->anzelem; ++i)
        {
            el[i] = numVertices;
            tl[i] = typeList[i];
            for (j = 0; j < rflFile->element[i].anznodes; j++)
            {
                // evtl doch bei 0 startend...
                vl[el[i] + j] = rflFile->element[i].nodes[j] - 1;
            }
            numVertices += rflFile->element[i].anznodes;
        }
        for (i = 0; i < rflFile->anznodes; ++i)
        {
            x_c[i] = (float)rflFile->node[i].x;
            y_c[i] = (float)rflFile->node[i].y;
            z_c[i] = (float)rflFile->node[i].z;
        }
        gridPort->setCurrentObject(gridObject);

        for (i = 0; i < NUMPORTS; i++)
        {
            int currentChoice = dofs[i]->getValue();
            if (currentChoice > 0)
            {
                if (currentChoice == 1)
                {
                    DOFROOT *vec[3] = { NULL, NULL, NULL };
                    DOFROOT *current = rflFile->dofroot;

                    while (current)
                    {
                        if (!current->exdof)
                        {
                            switch (current->typ)
                            {
                            case 10: // VX
                                vec[0] = current;
                                break;
                            case 11: // VY
                                vec[1] = current;
                                break;
                            case 12: // VZ
                                vec[2] = current;
                                break;
                            }
                        }
                        current = current->next;
                    }
                    // vec[...] zeigt jetzt auf x,y,z

                    coDoVec3 *dataObj;
                    if (numTimesteps->getValue() > 1)
                    {
                        char *objectName = new char[strlen(data[i]->getObjName()) + 50];
                        sprintf(objectName, "%s_%d", data[i]->getObjName(), timestep);
                        dataObj = new coDoVec3(objectName, vec[0]->anz);
                        dataObjects[i][numTimestepsRead] = dataObj;
                    }
                    else
                    {
                        dataObj = new coDoVec3(data[i]->getObjName(), vec[0]->anz);
                    }
                    float *u, *v, *w;
                    dataObj->getAddresses(&u, &v, &w);
                    for (j = 0; j < vec[0]->anz; j++)
                    {
                        u[j] = (float)(vec[0]->data[j]);
                        v[j] = (float)(vec[1]->data[j]);
                        w[j] = (float)(vec[2]->data[j]);
                    }
                    data[i]->setCurrentObject(dataObj);
                }
                else
                {
                    DOFROOT *current = rflFile->dofroot;
                    j = 0;
                    for (j = 0; j < (currentChoice - 1); j++)
                    {
                        current = current->next;
                        if (!current)
                            break;
                    }
                    if (current)
                    {

                        coDoFloat *dataObj;
                        if (numTimesteps->getValue() > 1)
                        {
                            char *objectName = new char[strlen(data[i]->getObjName()) + 50];
                            sprintf(objectName, "%s_%d", data[i]->getObjName(), timestep);
                            dataObj = new coDoFloat(objectName, current->anz);
                            dataObjects[i][numTimestepsRead] = dataObj;
                        }
                        else
                        {
                            dataObj = new coDoFloat(data[i]->getObjName(), current->anz);
                        }
                        float *d;
                        dataObj->getAddress(&d);
                        for (j = 0; j < current->anz; j++)
                        {
                            d[j] = (float)(current->data[j]);
                        }
                        data[i]->setCurrentObject(dataObj);
                    }
                }
            }
        }
        delete[] typeList;

        numTimestepsRead++;
        timestep++;
        timestep += numSkip->getValue();
    }
    if (numTimesteps->getValue() > 1)
    {

        coDoSet *mySet = new coDoSet(gridPort->getObjName(), grids);
        mySet->addAttribute("TIMESTEP", "1 1000");
        gridPort->setCurrentObject(mySet);
        for (i = 0; i < NUMPORTS; i++)
        {
            if (dataObjects[i][0] != NULL)
            {
                mySet = new coDoSet(data[i]->getObjName(), dataObjects[i]);
                data[i]->setCurrentObject(mySet);
            }
        }
    }

    for (j = 0; j < numTimestepsRead; j++)
    {
        delete grids[j];
    }
    delete[] grids;
    for (i = 0; i < NUMPORTS; i++)
    {
        for (j = 0; j < numTimestepsRead; j++)
        {
            delete dataObjects[i][j];
        }
        delete[] dataObjects[i];
    }
    delete[] dataObjects;

    return SUCCESS;
}

MODULE_MAIN(Reader, ReadRFL)
