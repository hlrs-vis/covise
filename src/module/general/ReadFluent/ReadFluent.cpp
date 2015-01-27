/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

/************************************************************************/

#ifdef WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/types.h>
#endif
#include <appl/ApplInterface.h>
#include "ReadFluent.h"
#include <api/coStepFile.h>

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>

#include <sys/stat.h>

int bigEndian = 1;

Fluent::Fluent(int argc, char *argv[])
    : coModule(argc, argv, "Read Fluent data")
{
    //"File reader for formatted Fluent(R) files V 0.99b"
    char *ChoiceInitVal[] = { (char *)"(none)" };
    // parameters
    p_casepath = addFileBrowserParam("CasePath", "filename");
    p_casepath->setValue("data", "*.cas");

    p_datapath = addFileBrowserParam("DataPath", "filename");
    p_datapath->setValue("data", "*.dat");

    p_data[0] = addChoiceParam("select_data_1", "Select Output Data");
    p_data[0]->setValue(1, ChoiceInitVal, 0);

    p_data[1] = addChoiceParam("select_data_2", "Select Output Data");
    p_data[1]->setValue(1, ChoiceInitVal, 0);

    p_data[2] = addChoiceParam("select_data_3", "Select Output Data");
    p_data[2]->setValue(1, ChoiceInitVal, 0);

    p_timesteps = addInt32Param("timesteps", "timesteps");
    p_timesteps->setValue(1);

    pFileIncrement = addInt32Param("file_increment", "increment in file numbers");
    pFileIncrement->setValue(1);

    p_skip = addInt32Param("skipped_files", "number of skipped files for each timestep");
    p_skip->setValue(0);

    p_outPort1 = addOutputPort("grid", "StructuredGrid|UnstructuredGrid", "grid");
    p_outPort2 = addOutputPort("polygons", "Polygons", "geometry polygons");
    p_outPort3 = addOutputPort("data1", "Float|Vec3", "data1");
    p_outPort4 = addOutputPort("data2", "Float|Vec3", "data2");
    p_outPort5 = addOutputPort("data3", "Float|Vec3", "data3");
    p_linePort = addOutputPort("lines", "Lines", "lines");

    numNodes = numFaces = numCells = 0;
    x_coords = NULL;
    y_coords = NULL;
    z_coords = NULL;
    vertices = NULL;
    typelist = NULL;
    facetype = NULL;
    rightNeighbor = NULL;
    leftNeighbor = NULL;
    cellvl = NULL;
    cellFlag = NULL;
    faceFlag = NULL;
    dataFileName = NULL;
    numVars = 0;
}

Fluent::~Fluent()
{
    // ...
}

//..........................................................................

void Fluent::param(const char *paramname, bool inMapLoading)
{
    const char *fileName = NULL;

    if (strcmp(p_casepath->getName(), paramname) == 0)
    {
        fileName = p_casepath->getValue();
        if (fileName == NULL)
        {
            sendError("ERROR: filename is NULL");
            return;
        }
        if (readFile(fileName) < 0)
            return;
        if (dataFileName)
        {
            if (!inMapLoading)
            {
                parseDat(dataFileName);
                updateChoice();
            }
        }
    }
    else if (strcmp(p_datapath->getName(), paramname) == 0)
    {
        fileName = p_datapath->getValue();
        if (fileName != NULL)
        {
            delete[] dataFileName;
            dataFileName = new char[strlen(fileName) + 1];
            strcpy(dataFileName, fileName);
            if (!inMapLoading)
            {
                parseDat(fileName);
                updateChoice();
            }
        }
        else
        {
            numVars = 0;
            updateChoice();
        }
    }
}

int Fluent::compute(const char *)
{
    int i, n, numWalls, dataSelection[3];
    float *x_c, *y_c, *z_c;
    int *vl, *pl, *el, *tl;

    char buf[100];
    char *next_path = NULL;
    elementTypeList = NULL;
    coDoPolygons *polygonObject; // output object
    coDoUnstructuredGrid *gridObject;
    coDoLines *lineObject;

    /*const char * gridObjectName = p_outPort1->getObjName();
   const char * polygonObjectName = p_outPort2->getObjName();
   const char * data1ObjectName = p_outPort3->getObjName();
   const char * data2ObjectName = p_outPort4->getObjName();
   const char * data3ObjectName = p_outPort5->getObjName();*/

    dataSelection[0] = p_data[0]->getValue();
    dataSelection[1] = p_data[1]->getValue();
    dataSelection[2] = p_data[2]->getValue();

    int timesteps = p_timesteps->getValue();

    if (timesteps < 1)
    {
        timesteps = 1;
        p_timesteps->setValue(timesteps);
    }

    int skip_value = p_skip->getValue();

    coDistributedObject **time_outputgrid;
    coDistributedObject **time_outputpolygon;
    coDistributedObject **time_outputdata1;
    coDistributedObject **time_outputdata2;
    coDistributedObject **time_outputdata3;

    coDistributedObject **time_outputlines;

    time_outputgrid = new coDistributedObject *[timesteps + 1];
    time_outputgrid[timesteps] = NULL;

    time_outputpolygon = new coDistributedObject *[timesteps + 1];
    time_outputpolygon[timesteps] = NULL;

    time_outputdata1 = new coDistributedObject *[timesteps + 1];
    time_outputdata1[timesteps] = NULL;

    time_outputdata2 = new coDistributedObject *[timesteps + 1];
    time_outputdata2[timesteps] = NULL;

    time_outputdata3 = new coDistributedObject *[timesteps + 1];
    time_outputdata3[timesteps] = NULL;

    time_outputlines = new coDistributedObject *[timesteps + 1];
    time_outputlines[timesteps] = 0;

    coStepFile *step_data = new coStepFile(dataFileName);
    step_data->set_skip_value(skip_value);

    int fileIncr(pFileIncrement->getValue());
    step_data->set_delta(fileIncr);

    // create the COVISE output object
    numVertices = 0;
    numWalls = 0;
    for (i = 0; i < numFaces; i++)
    {
        if ((facetype[i] == 3) || (facetype[i] == 1003)) // face is a wall
        {
            numWalls++;
            numVertices += typelist[i];
        }
    }

    if (timesteps > 1)
        sprintf(buf, "%s_0", p_outPort2->getObjName());
    else
        strcpy(buf, p_outPort2->getObjName());

    polygonObject = new coDoPolygons(buf, numNodes, numVertices, numWalls);
    polygonObject->getAddresses(&x_c, &y_c, &z_c, &vl, &pl);
    polygonObject->addAttribute("vertexOrder", "2");
    n = 0;
    numWalls = 0;
    for (i = 0; i < numFaces; i++)
    {
        if ((facetype[i] == 3) || (facetype[i] == 1003)) // face is a wall
        {
            pl[numWalls] = n;
            n += typelist[i];
            numWalls++;
        }
    }
    numWalls = 0;
    for (i = 0; i < numFaces; i++)
    {
        if ((facetype[i] == 3) || (facetype[i] == 1003)) // face is a wall
        {
            vl[pl[numWalls] + 2] = vertices[i * 4 + 2] - 1;
            vl[pl[numWalls] + 1] = vertices[i * 4 + 1] - 1;
            vl[pl[numWalls] + 0] = vertices[i * 4 + 0] - 1;
            if (typelist[i] == 4)
                vl[pl[numWalls] + 3] = vertices[i * 4 + 3] - 1;
            n += typelist[i];
            numWalls++;
        }
    }

    for (i = 0; i < numNodes; i++)
    {
        x_c[i] = x_coords[i];
        y_c[i] = y_coords[i];
        z_c[i] = z_coords[i];
    }
    //delete polygonObject;
    //p_outPort2->setCurrentObject(polygonObject);

    // map the cell indices to the element list
    // the array will be used in Application::readDat(..)
    elemToCells_ = new int[numCells];

    if (isTetOnly())
    {
        sendInfo("buliding Tetrahedra Mesh!");
        cellvl = new int[numCells * 4];

        for (i = 0; i < numCells * 4; i++)
            cellvl[i] = -1;

        for (i = 0; i < numFaces; i++)
        {
            if (typelist[i] != 3)
            {
                sendError("Sorry, only tetrahedra supported!");
                return STOP_PIPELINE;
            }
            if (rightNeighbor[i])
            {
                addTriangle(i, rightNeighbor[i] - 1);
            }
            if (leftNeighbor[i])
            {
                addTriangle(i, leftNeighbor[i] - 1);
            }
        }

        if (timesteps > 1)
            sprintf(buf, "%s_0", p_outPort1->getObjName());
        else
            strcpy(buf, p_outPort1->getObjName());

        gridObject = new coDoUnstructuredGrid(buf, numCells, numCells * 4, numNodes, 1);
        gridObject->getAddresses(&el, &vl, &x_c, &y_c, &z_c);
        gridObject->getTypeList(&tl);
        for (i = 0; i < numNodes; i++)
        {
            x_c[i] = x_coords[i];
            y_c[i] = y_coords[i];
            z_c[i] = z_coords[i];
        }
        for (i = 0; i < numCells; i++)
        {
            tl[i] = TYPE_TETRAHEDER;
            el[i] = 4 * i;
            elemToCells_[i] = i;
        }
        for (i = 0; i < numCells * 4; i++)
        {
            vl[i] = cellvl[i];
        }
        //delete gridObject;
        //p_outPort1->setCurrentObject(gridObject);
    }
    else
    {
        sendInfo("buliding mixed element mesh!");
        //freeElements.noDelete = 1;
        elementTypeList = new int[numCells];
        Elements = new Element *[numCells];
        for (i = 0; i < numCells; i++)
            Elements[i] = NULL;
        cellvl = new int[numCells * 8];
        newElementList = new int[numCells];

        for (i = 0; i < numCells * 8; i++)
        {
            if (i < numCells)
            {
                newElementList[i] = 0;
                elemToCells_[i] = -1;
            }
            cellvl[i] = -1;
        }
        numVertices = 0;
        numElements = 0;
        numFreeElements = 0;
        numFreeAlloc = 1000;
        freeElements = new Element *[numFreeAlloc];

        for (i = 0; i < numFreeAlloc; ++i)
            freeElements[i] = NULL;

        for (i = 0; i < numFaces; i++)
        {
            if ((rightNeighbor[i]) && (cellFlag[rightNeighbor[i]] == 0))
            {
                addFace(i, rightNeighbor[i] - 1, 1);
            }
            if ((leftNeighbor[i]) && (cellFlag[leftNeighbor[i]] == 0))
            {
                addFace(i, leftNeighbor[i] - 1, 0);
            }
        }

        int unknwnEle = 0;
        for (i = 0; i < numElements; ++i)
        {
            if (Elements[i] != (Element *)0x1)
            {
                unknwnEle++;
                //Elements[i]->info();
                //Elements[i]->checkTet();
            }
        }
        //	cerr << "Application::compute(..) --> numFaces: " <<  numFaces << endl;

        for (i = 0; i <= numFreeElements; i++)
        {
            delete freeElements[i];
        }
        delete[] freeElements;

        for (i = 0; i < numElements; ++i)
        {
            if (Elements[i] != (Element *)0x1)
            {
                delete Elements[i];
            }
        }
        delete[] Elements;

        /*
      freeElements.noDelete = 0;
      freeElements.reset();
      while(freeElements.current()!=NULL)
      {
          freeElements.remove();
      }*/
        if (timesteps > 1)
            sprintf(buf, "%s_0", p_outPort1->getObjName());
        else
            strcpy(buf, p_outPort1->getObjName());

        gridObject = new coDoUnstructuredGrid(buf, numElements, numVertices, numNodes, 1);
        gridObject->getAddresses(&el, &vl, &x_c, &y_c, &z_c);
        gridObject->getTypeList(&tl);
        for (i = 0; i < numNodes; i++)
        {
            x_c[i] = x_coords[i];
            y_c[i] = y_coords[i];
            z_c[i] = z_coords[i];
        }
        //int elIndex = 0;
        for (i = 0; i < numElements; i++)
        {
            tl[i] = elementTypeList[i];
            el[i] = newElementList[i];
            //elIndex+=UnstructuredGrid_Num_Nodes[elementTypeList[i]];
        }
        for (i = 0; i < numVertices; i++)
        {
            vl[i] = cellvl[i];
        }
        //delete gridObject;
        //p_outPort1->setCurrentObject(gridObject);
        delete[] newElementList;
        delete[] cellvl;
    }

    lineObject = makeLines();

    int numSteps = 0;

    if (numVars == 0)
    {
        if ((dataSelection[0] != 0) || (dataSelection[1] != 0) || (dataSelection[2] != 0))
            parseDat(dataFileName);
    }

    for (i = 0; i < timesteps; i++)
    {
        step_data->get_nextpath(&next_path);

        if (next_path)
        {
            if (dataSelection[0] != 0)
            {
                if (timesteps > 1)
                    sprintf(buf, "%s_%d", p_outPort3->getObjName(), numSteps);
                else
                    strcpy(buf, p_outPort3->getObjName());

                if (varIsFace[dataSelection[0] - 1])
                    time_outputdata1[numSteps] = createFaceDataObject(buf, varTypes[dataSelection[0] - 1], next_path);
                else
                    time_outputdata1[numSteps] = createDataObject(buf, varTypes[dataSelection[0] - 1], next_path);

                //p_outPort3->setCurrentObject(time_outputdata1[i]);
            }
            if (dataSelection[1] != 0)
            {
                if (timesteps > 1)
                    sprintf(buf, "%s_%d", p_outPort4->getObjName(), numSteps);
                else
                    strcpy(buf, p_outPort4->getObjName());

                if (varIsFace[dataSelection[1] - 1])
                    time_outputdata2[numSteps] = createFaceDataObject(buf, varTypes[dataSelection[1] - 1], next_path);
                else
                    time_outputdata2[numSteps] = createDataObject(buf, varTypes[dataSelection[1] - 1], next_path);
                //p_outPort4->setCurrentObject(time_outputdata2[i]);
            }
            if (dataSelection[2] != 0)
            {
                if (timesteps > 1)
                    sprintf(buf, "%s_%d", p_outPort5->getObjName(), numSteps);
                else
                    strcpy(buf, p_outPort5->getObjName());

                if (varIsFace[dataSelection[2] - 1])
                    time_outputdata3[numSteps] = createFaceDataObject(buf, varTypes[dataSelection[2] - 1], next_path);
                else
                    time_outputdata3[numSteps] = createDataObject(buf, varTypes[dataSelection[2] - 1], next_path);
                //p_outPort5->setCurrentObject(time_outputdata3[i]);
            }
            numSteps++;
        } //if next_path
    } //for timesteps

    delete step_data;

    if (timesteps > 1)
    {
        for (i = 0; i < numSteps; i++)
        {
            time_outputgrid[i] = gridObject;
            gridObject->incRefCount();
        }
        time_outputgrid[numSteps] = NULL;
        coDoSet *time_grd = new coDoSet(p_outPort1->getObjName(), time_outputgrid);
        sprintf(buf, "1 %d", numSteps);
        time_grd->addAttribute("TIMESTEP", buf);

        delete gridObject;
        delete[] time_outputgrid;

        p_outPort1->setCurrentObject(time_grd);

        for (i = 0; i < numSteps; i++)
        {
            time_outputpolygon[i] = polygonObject;
            polygonObject->incRefCount();
        }
        time_outputpolygon[numSteps] = NULL;
        coDoSet *time_pol = new coDoSet(p_outPort2->getObjName(), time_outputpolygon);
        sprintf(buf, "1 %d", numSteps);
        time_pol->addAttribute("TIMESTEP", buf);

        delete polygonObject;
        delete[] time_outputpolygon;

        p_outPort2->setCurrentObject(time_pol);

        if (dataSelection[0] != 0)
        {
            time_outputdata1[numSteps] = NULL;
            coDoSet *time_data1 = new coDoSet(p_outPort3->getObjName(), time_outputdata1);
            sprintf(buf, "1 %d", numSteps);
            time_data1->addAttribute("TIMESTEP", buf);

            for (i = 0; i < numSteps; i++)
                delete time_outputdata1[i];
            delete[] time_outputdata1;

            p_outPort3->setCurrentObject(time_data1);
        }

        if (dataSelection[1] != 0)
        {
            time_outputdata2[numSteps] = NULL;
            coDoSet *time_data2 = new coDoSet(p_outPort4->getObjName(), time_outputdata2);
            sprintf(buf, "1 %d", numSteps);
            time_data2->addAttribute("TIMESTEP", buf);

            for (i = 0; i < numSteps; i++)
                delete time_outputdata2[i];
            delete[] time_outputdata2;

            p_outPort4->setCurrentObject(time_data2);
        }

        if (dataSelection[2] != 0)
        {
            time_outputdata3[numSteps] = NULL;
            coDoSet *time_data3 = new coDoSet(p_outPort5->getObjName(), time_outputdata3);
            sprintf(buf, "1 %d", numSteps);
            time_data3->addAttribute("TIMESTEP", buf);

            for (i = 0; i < numSteps; i++)
                delete time_outputdata3[i];
            delete[] time_outputdata3;

            p_outPort5->setCurrentObject(time_data3);
        }
    }
    else
    {
        p_outPort1->setCurrentObject(gridObject);
        p_outPort2->setCurrentObject(polygonObject);
        p_linePort->setCurrentObject(lineObject);

        if (dataSelection[0] != 0)
            p_outPort3->setCurrentObject(time_outputdata1[0]);
        if (dataSelection[1] != 0)
            p_outPort4->setCurrentObject(time_outputdata2[0]);
        if (dataSelection[2] != 0)
            p_outPort5->setCurrentObject(time_outputdata3[0]);
    }
    delete[] elemToCells_;
    delete[] elementTypeList;
    return CONTINUE_PIPELINE;
}

coDistributedObject *Fluent::createDataObject(const char *name, int dataSelection, char *dataFileName)
{
    coDoVec3 *VObject;
    coDoFloat *SObject;
    float *u, *v, *w;
    if (dataSelection < 0)
    {
        VObject = new coDoVec3((char *)name, numCells);
        VObject->getAddresses(&u, &v, &w);
        readDat(dataFileName, u, VectorScalarMap[-dataSelection], READ_CELLS);
        readDat(dataFileName, v, VectorScalarMap[-dataSelection] + 1, READ_CELLS);
        readDat(dataFileName, w, VectorScalarMap[-dataSelection] + 2, READ_CELLS);
        return VObject;
        //delete VObject;
    }
    else
    {

        if (dataSelection == 2)
        {
            // we have momentum data
            VObject = new coDoVec3((char *)name, numCells);
            VObject->getAddresses(&u, &v, &w);
            readDat(dataFileName, u, dataSelection, READ_CELLS, v, w);
            return VObject;
            //delete VObject;
        }
        else
        {
            SObject = new coDoFloat((char *)name, numCells);
            SObject->getAddress(&u);
            readDat(dataFileName, u, dataSelection, READ_CELLS);
            return SObject;
            //delete SObject;
        }
    }
}

coDistributedObject *Fluent::createFaceDataObject(const char *name, int dataSelection, char *dataFileName)
{
    coDoVec3 *VObject;
    coDoFloat *SObject;
    float *u, *v, *w;
    if (dataSelection < 0)
    {
        VObject = new coDoVec3((char *)name, numNodes);
        VObject->getAddresses(&u, &v, &w);
        readDat(dataFileName, u, VectorScalarMap[-dataSelection], READ_FACES);
        readDat(dataFileName, v, VectorScalarMap[-dataSelection] + 1, READ_FACES);
        readDat(dataFileName, w, VectorScalarMap[-dataSelection] + 2, READ_FACES);
        return VObject;
    }
    else
    {
        if (dataSelection == 2)
        {
            // we have momentum data
            VObject = new coDoVec3((char *)name, numNodes);
            VObject->getAddresses(&u, &v, &w);
            readDat(dataFileName, u, dataSelection, READ_FACES, v, w);
            return VObject;
        }
        else
        {
            SObject = new coDoFloat((char *)name, numNodes);
            SObject->getAddress(&u);
            readDat(dataFileName, u, dataSelection, READ_FACES);
            return SObject;
        }
    }
}

Element::Element()
    : bad_(0)
    , empty_(1)
    , numTriangles(0)
    , numQuads(0)
{
}

Element::Element(int cell)
    : cell_(cell)
    , bad_(0)
    , empty_(1)
    , numTriangles(0)
    , numQuads(0)
{
}

void
Element::badInfo()
{
    if ((numTriangles > 4) || (numQuads > 6))
    {
        fprintf(stderr, "Element # %d numTriang: %d     numQuads: %d\n",
                cell_, numTriangles, numQuads);

        int i;
        for (i = 0; i < numTriangles; ++i)
            fprintf(stderr, "%d + %d + %d :", triangles[i][0], triangles[i][1], triangles[i][2]);

        for (i = 0; i < numQuads; ++i)
            fprintf(stderr, "%d - %d - %d -%d :", quads[i][0], quads[i][1], quads[i][2], quads[i][3]);

        fprintf(stderr, "\n*****************\n");
    }
}

Element::~Element()
{
}

void Element::reset()
{
    numQuads = 0;
    numTriangles = 0;
    empty_ = 1;
}

int
Element::getQuad(int v1, int v2, int notNum)
{
    int i, n;
    for (i = 0; i < numQuads; i++)
    {
        if (i != notNum)
        {
            for (n = 0; n < 4; n++)
            {
                if (quads[i][n] == v1)
                {
                    if (n > 0)
                    {
                        if (quads[i][n - 1] == v2)
                            return i;
                        if (n == 3)
                        {
                            if (quads[i][0] == v2)
                                return i;
                        }
                        else
                        {
                            if (quads[i][n + 1] == v2)
                                return i;
                        }
                    }
                    else
                    {
                        if (quads[i][3] == v2)
                            return i;
                        if (quads[i][1] == v2)
                            return i;
                    }
                }
            }
        }
    }
    return -1;
}

void Element::setVertices(int *vl, int &numVertices)
{ // add Vertices to the Vertex list
    switch (getType())
    {
    case TYPE_TETRAHEDER:
    {
        int v;
        vl[numVertices] = triangles[0][0];
        vl[numVertices + 1] = triangles[0][1];
        vl[numVertices + 2] = triangles[0][2];
        v = triangles[1][0];
        if ((vl[numVertices] != v) && (vl[numVertices + 1] != v) && (vl[numVertices + 2] != v))
        {
            vl[numVertices + 3] = v;
        }
        else
        {
            v = triangles[1][1];
            if ((vl[numVertices] != v) && (vl[numVertices + 1] != v) && (vl[numVertices + 2] != v))
            {
                vl[numVertices + 3] = v;
            }
            else
            {
                v = triangles[1][2];
                if ((vl[numVertices] != v) && (vl[numVertices + 1] != v) && (vl[numVertices + 2] != v))
                {
                    vl[numVertices + 3] = v;
                }
                else
                {
                    vl[numVertices + 3] = 0;
                    cerr << " Malformed Tetraheder " << endl;
                }
            }
        }
        // Do we have to turn it inside out?
        /*float p0[3],p1[3],p2[3],p3[3];
         p0[0]=x_coords[vl[numVertices]];
         p0[1]=x_coords[vl[numVertices]];
         p0[2]=x_coords[vl[numVertices]];
         p1[0]=x_coords[vl[numVertices+1]];
         p1[1]=x_coords[vl[numVertices+1]];
         p1[2]=x_coords[vl[numVertices+1]];
         p2[0]=x_coords[vl[numVertices+2]];
         p2[1]=x_coords[vl[numVertices+2]];
         p2[2]=x_coords[vl[numVertices+2]];
         p3[0]=x_coords[vl[numVertices+3]];
         p3[1]=x_coords[vl[numVertices+3]];
         p3[2]=x_coords[vl[numVertices+3]];
         if(tetraVol(p0,p1,p2,p3)<0)
         {
         v=vl[numVertices+2];
         vl[numVertices+2]=vl[numVertices];
         vl[numVertices]=v;
         }*/
        numVertices += 4;
    }
    break;
    case TYPE_PYRAMID:
    {

        vl[numVertices] = quads[0][0];
        numVertices++;
        vl[numVertices] = quads[0][1];
        numVertices++;
        vl[numVertices] = quads[0][2];
        numVertices++;
        vl[numVertices] = quads[0][3];
        numVertices++;
        int v = triangles[0][0];
        if ((vl[numVertices - 1] != v) && (vl[numVertices - 2] != v) && (vl[numVertices - 3] != v) && (vl[numVertices - 4] != v))
        {
            vl[numVertices] = v;
        }
        else
        {
            v = triangles[0][1];
            if ((vl[numVertices - 1] != v) && (vl[numVertices - 2] != v) && (vl[numVertices - 3] != v) && (vl[numVertices - 4] != v))
            {
                vl[numVertices] = v;
            }
            else
            {
                v = triangles[0][2];
                if ((vl[numVertices - 1] != v) && (vl[numVertices - 2] != v) && (vl[numVertices - 3] != v) && (vl[numVertices - 4] != v))
                {
                    vl[numVertices] = v;
                }
                else
                {
                    vl[numVertices] = 0;
                    cerr << " Malformed Pyramid " << endl;
                }
            }
        }
        numVertices++;
    }
    break;
    case TYPE_PRISM:
    {

        int neighborQuad, oppVertex, oppV2;
        neighborQuad = getQuad(triangles[0][0], triangles[0][1], -1);
        if (neighborQuad < 0)
        {
            break;
        }
        vl[numVertices] = triangles[0][0];
        numVertices++;
        vl[numVertices] = triangles[0][1];
        numVertices++;
        vl[numVertices] = triangles[0][2];
        numVertices++;
        if (quads[neighborQuad][0] == triangles[0][0])
        {
            if (quads[neighborQuad][1] == triangles[0][1])
            {
                oppVertex = quads[neighborQuad][3];
                oppV2 = quads[neighborQuad][2];
            }
            else
            {
                oppVertex = quads[neighborQuad][1];
                oppV2 = quads[neighborQuad][2];
            }
        }
        else if (quads[neighborQuad][1] == triangles[0][0])
        {
            if (quads[neighborQuad][2] == triangles[0][1])
            {
                oppVertex = quads[neighborQuad][0];
                oppV2 = quads[neighborQuad][3];
            }
            else
            {
                oppVertex = quads[neighborQuad][2];
                oppV2 = quads[neighborQuad][3];
            }
        }
        else if (quads[neighborQuad][2] == triangles[0][0])
        {
            if (quads[neighborQuad][3] == triangles[0][1])
            {
                oppVertex = quads[neighborQuad][1];
                oppV2 = quads[neighborQuad][0];
            }
            else
            {
                oppVertex = quads[neighborQuad][3];
                oppV2 = quads[neighborQuad][0];
            }
        }
        else if (quads[neighborQuad][3] == triangles[0][0])
        {
            if (quads[neighborQuad][0] == triangles[0][1])
            {
                oppVertex = quads[neighborQuad][2];
                oppV2 = quads[neighborQuad][1];
            }
            else
            {
                oppVertex = quads[neighborQuad][0];
                oppV2 = quads[neighborQuad][1];
            }
        }
        else
            break;
        vl[numVertices] = oppVertex;
        numVertices++;
        vl[numVertices] = oppV2;
        numVertices++;
        if ((triangles[1][0] != oppVertex) && (triangles[1][0] != oppV2))
            vl[numVertices] = triangles[1][0];
        else if ((triangles[1][1] != oppVertex) && (triangles[1][1] != oppV2))
            vl[numVertices] = triangles[1][1];
        else if ((triangles[1][2] != oppVertex) && (triangles[1][2] != oppV2))
            vl[numVertices] = triangles[1][2];
        numVertices++;
    }
    break;
    case TYPE_HEXAEDER:
    {
        int neighborQuad, oppositeQuad, oppVertex;
        neighborQuad = getQuad(quads[0][0], quads[0][1], 0);
        if (neighborQuad < 0)
        {
            break;
        }
        vl[numVertices] = quads[0][0];
        numVertices++;
        vl[numVertices] = quads[0][1];
        numVertices++;
        vl[numVertices] = quads[0][2];
        numVertices++;
        vl[numVertices] = quads[0][3];
        numVertices++;

        // gegenueberliegenden Vertex und gegenueberliegendes Viereck suchen
        if (quads[neighborQuad][0] == quads[0][0])
        {
            if (quads[neighborQuad][1] == quads[0][1])
            {
                oppVertex = quads[neighborQuad][3];
                //oppVertex2= quads[neighborQuad][2];
                oppositeQuad = getQuad(quads[neighborQuad][2], quads[neighborQuad][3], neighborQuad);
            }
            else
            {
                oppVertex = quads[neighborQuad][1];
                // oppVertex2= quads[neighborQuad][2];
                oppositeQuad = getQuad(quads[neighborQuad][2], quads[neighborQuad][1], neighborQuad);
            }
        }
        else if (quads[neighborQuad][1] == quads[0][0])
        {
            if (quads[neighborQuad][2] == quads[0][1])
            {
                oppVertex = quads[neighborQuad][0];
                //oppVertex2= quads[neighborQuad][3];
                oppositeQuad = getQuad(quads[neighborQuad][0], quads[neighborQuad][3], neighborQuad);
            }
            else
            {
                //oppVertex2= quads[neighborQuad][3];
                oppVertex = quads[neighborQuad][2];
                oppositeQuad = getQuad(quads[neighborQuad][2], quads[neighborQuad][3], neighborQuad);
            }
        }
        else if (quads[neighborQuad][2] == quads[0][0])
        {
            if (quads[neighborQuad][3] == quads[0][1])
            {
                oppVertex = quads[neighborQuad][1];
                oppositeQuad = getQuad(quads[neighborQuad][0], quads[neighborQuad][1], neighborQuad);
            }
            else
            {
                oppVertex = quads[neighborQuad][3];
                oppositeQuad = getQuad(quads[neighborQuad][3], quads[neighborQuad][0], neighborQuad);
            }
        }
        else if (quads[neighborQuad][3] == quads[0][0])
        {
            if (quads[neighborQuad][0] == quads[0][1])
            {
                oppVertex = quads[neighborQuad][2];
                oppositeQuad = getQuad(quads[neighborQuad][1], quads[neighborQuad][2], neighborQuad);
            }
            else
            {
                oppVertex = quads[neighborQuad][0];
                oppositeQuad = getQuad(quads[neighborQuad][0], quads[neighborQuad][1], neighborQuad);
            }
        }
        else
            break;
        int i;
        if (oppositeQuad < 0)
        {
            numVertices -= 4;
            break;
        }
        for (i = 0; i < 4; i++) // gegenueberliegenden Vertex im gegenueberliegenden Viereck suchen
        {
            if (quads[oppositeQuad][i] == oppVertex)
                break;
        }
        if (i == 4)
        {
            cerr << "Internal Error" << endl;
        }
        vl[numVertices] = oppVertex;
        numVertices++;
        i--;
        if (i < 0)
            i = 3;
        vl[numVertices] = quads[oppositeQuad][i];
        numVertices++;
        i--;
        if (i < 0)
            i = 3;
        vl[numVertices] = quads[oppositeQuad][i];
        numVertices++;
        i--;
        if (i < 0)
            i = 3;
        vl[numVertices] = quads[oppositeQuad][i];
        numVertices++;
    }
    break;
    default:
    {
    }
    break;
    }
}

int Element::getType()
{

    if (numQuads == 6)
    {
        // this might be a Hexaeder, have a closer look
        // there might be a face that belongs to a nonconformal interface, this would have to be removed
        // check, if one of the quads vertices are hanging nodes (do not occur in any of the other quads
        /*int i,j,k,l;
      for(i=0;i<numQuads;i++)
      {
          for(j=0;j<4;j++)
          {

              for(k=0;k<numQuads;k++)
              {
                  if(k!=i)
                  {
                      for(l=0;l<4;l++)
      {
      if(quads[i][j]==quads[k][l])
      {
      k=numQuads+1;
      break;
      }
      }
      }
      }
      if(k == numQuads)
      { // we did not find this vertex in any of the other quads, so remove this quad
      i = numQuads+1;
      for(j=i+1;j<numQuads;j++)
      {
      for(l=0;l<4;l++)
      {
      quads[j-1][l] = quads[j][l];
      }
      qside[j-1] = qside[j];
      }
      numQuads--;
      break;
      }
      }
      }	  */
        if (numQuads == 6)
            return TYPE_HEXAEDER;
    }
    if (numTriangles == 4)
    {
        int vertices[50];
        int numv = 0, i, j, n;
        // we check if we have a closed polygon
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < 3; j++)
            {
                for (n = 0; n < numv; n++)
                {
                    if (vertices[n] == triangles[i][j])
                        break;
                }
                if (n == numv)
                {
                    vertices[numv] = triangles[i][j];
                    numv++;
                }
            }
        }
        if (numv == 4)
            return TYPE_TETRAHEDER;
        else
            return -1;
        // this might be a Hexaeder, have a closer look
        // there might be a face that belongs to a nonconformal interface, this would have to be removed
        // check, if one of the quads vertices are hanging nodes (do not occur in any of the other quads
        //int i,j,k,l;
        /*for(i=0;i<numQuads;i++)
      {
          for(j=0;j<4;j++)
          {

              for(k=0;k<numTriangles;k++)
              {
                  if(k!=i)
                  {
                      for(l=0;l<3;l++)
                      {
      if(quads[i][j]==triangles[k][l])
      {
      k=numTriangles+1;
      break;
      }
      }
      }
      }
      if(k == numQuads)
      { // we did not find this vertex in any of the triangles, so remove this quad
      for(j=i+1;j<numQuads;j++)
      {
      for(l=0;l<4;l++)
      {
      quads[j-1][l] = quads[j][l];
      }
      qside[j-1] = qside[j];
      }
      numQuads--;

      if(numQuads)
      i--;
      break;
      }
      }
      }*/
        if (numQuads == 1)
            return TYPE_PYRAMID;
    }

    if ((numQuads == 3) && (numTriangles == 2))
        return TYPE_PRISM;

    return -1;
}

int Element::checkTet()
{
    if (numTriangles > 4)
    {
        if (numQuads == 0)
        {
            int vertices[maxNumTriang * 3];
            int numv = 0, i, j, n;
            for (i = 0; i < maxNumTriang * 3; ++i)
                vertices[i] = -1;
            int numTet = 0;
            // we check if we have a closed polygon
            for (i = 0; i < numTriangles; i++)
            {
                for (j = 0; j < 3; j++)
                {
                    for (n = 0; n < numv; n++)
                    {
                        if (vertices[n] == triangles[i][j])
                            break;
                    }
                    if (n == numv)
                    {
                        vertices[numv] = triangles[i][j];
                        numv++;
                    }
                    if (numv == 4)
                    {
                        numTet++;
                        numv = 0;
                        int ii;
                        fprintf(stderr, "   TET: ");
                        for (ii = 0; ii < 4; ++ii)
                        {
                            fprintf(stderr, " -- %d", vertices[ii]);
                            vertices[ii] = -1;
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
            if (numTet > 0)
            {

                fprintf(stderr, "Element::checkTet() found Tetrahedron   ->%d<-\n ", numTet);
                return TYPE_TETRAHEDER;
            }
            else
                return -1;
        }
    }
    return 1;
}

int Element::addFace(int *vertices, int face, int type, int side)
{

    if ((type != 3) && (type != 4) && (type != 2))
    {
        fprintf(stderr, "Element::addFace(..) got unknown type of face: %d\n", type);
        return 0;
    }
    int *v = vertices + (face * 4);

    if (side) // right side
    {
        if (type == 3)
        {
            triangles[numTriangles][0] = v[0] - 1;
            triangles[numTriangles][1] = v[1] - 1;
            triangles[numTriangles][2] = v[2] - 1;
            numTriangles++;
        }
        if (type == 4)
        {
            qside[numQuads] = side;
            quads[numQuads][0] = v[0] - 1;
            quads[numQuads][1] = v[1] - 1;
            quads[numQuads][2] = v[2] - 1;
            quads[numQuads][3] = v[3] - 1;
            numQuads++;
        }
    }
    else // left side
    {
        if (type == 3)
        {
            triangles[numTriangles][0] = v[2] - 1;
            triangles[numTriangles][1] = v[1] - 1;
            triangles[numTriangles][2] = v[0] - 1;
            numTriangles++;
        }
        if (type == 4)
        {
            qside[numQuads] = side;
            quads[numQuads][0] = v[0] - 1;
            quads[numQuads][1] = v[3] - 1;
            quads[numQuads][2] = v[2] - 1;
            quads[numQuads][3] = v[1] - 1;
            numQuads++;
        }
    }

    if (getType() > 0)
        return 1;
    return 0;
}

void
Fluent::addFace(int face, int cell, int side)
{

    if (faceFlag[face] != 0)
        return;

    if (typelist[face] == 2)
        return;

    // debug
    /*if(cell == 36)
   {
       int i;
       fprintf(stderr, "Side: %2d Face: %4d ",side,face);
       for (i=0;i<typelist[face];i++)
       {
           fprintf(stderr," %4d ",vertices[face*4+i]-1);
       }
       fprintf(stderr,"\n");
   }*/

    if (Elements[cell] == NULL)
    {
        //if(freeElements.num())
        if (numFreeElements)
        {
            //freeElements.reset();
            //Elements[cell] = freeElements.current();
            //freeElements.remove();
            numFreeElements--;
            Elements[cell] = freeElements[numFreeElements];
        }
        else
        {
            Elements[cell] = new Element(cell);
        }
    }
    if (Elements[cell] == (Element *)0x1)
    {
        cerr << "Reused complete Element\n";
        return;
    }
    if (Elements[cell])
    {
        // if this returns true, we have a complete Element
        if (Elements[cell]->addFace(vertices, face, typelist[face], side))
        {
            elementTypeList[numElements] = Elements[cell]->getType();
            newElementList[numElements] = numVertices;
            elemToCells_[cell] = numElements;
            Elements[cell]->setVertices(cellvl, numVertices);
            Elements[cell]->reset();
            if (numFreeElements >= numFreeAlloc)
            {
                Element **oldFreeElements = freeElements;
                freeElements = new Element *[numFreeAlloc * 2];
                memcpy(freeElements, oldFreeElements, numFreeElements * sizeof(Element *));
                int ii;
                for (ii = numFreeElements + 1; ii < numFreeAlloc * 2; ++ii)
                    freeElements[ii] = NULL;
                numFreeAlloc *= 2;
            }
            freeElements[numFreeElements] = Elements[cell];
            numFreeElements++;
            //freeElements.append(Elements[cell]);

            Elements[cell] = (Element *)0x1;
            numElements++;
        }
    }
    else
    {
        cerr << "Out or Memory in addFace" << endl;
        return;
    }
}

void Fluent::addTriangle(int face, int cell)
{
    int ci = cell * 4;
    int v;
    if (cellvl[ci] == -1) // first triangle of Tetr.
    {
        cellvl[ci] = vertices[face * 4] - 1;
        cellvl[ci + 1] = vertices[face * 4 + 1] - 1;
        cellvl[ci + 2] = vertices[face * 4 + 2] - 1;
    }
    else if (cellvl[ci + 3] == -1)
    {
        v = vertices[face * 4] - 1;
        if ((cellvl[ci] != v) && (cellvl[ci + 1] != v) && (cellvl[ci + 2] != v))
        {
            cellvl[ci + 3] = v;
        }
        else
        {
            v = vertices[face * 4 + 1] - 1;
            if ((cellvl[ci] != v) && (cellvl[ci + 1] != v) && (cellvl[ci + 2] != v))
            {
                cellvl[ci + 3] = v;
            }
            else
            {
                v = vertices[face * 4 + 2] - 1;
                if ((cellvl[ci] != v) && (cellvl[ci + 1] != v) && (cellvl[ci + 2] != v))
                {
                    cellvl[ci + 3] = v;
                }
            }
        }
        // Do we have to turn it inside out?
        float p0[3], p1[3], p2[3], p3[3];
        p0[0] = x_coords[cellvl[ci]];
        p0[1] = x_coords[cellvl[ci]];
        p0[2] = x_coords[cellvl[ci]];
        p1[0] = x_coords[cellvl[ci + 1]];
        p1[1] = x_coords[cellvl[ci + 1]];
        p1[2] = x_coords[cellvl[ci + 1]];
        p2[0] = x_coords[cellvl[ci + 2]];
        p2[1] = x_coords[cellvl[ci + 2]];
        p2[2] = x_coords[cellvl[ci + 2]];
        p3[0] = x_coords[cellvl[ci + 3]];
        p3[1] = x_coords[cellvl[ci + 3]];
        p3[2] = x_coords[cellvl[ci + 3]];
        if (tetraVol(p0, p1, p2, p3) < 0)
        {
            v = cellvl[ci + 2];
            cellvl[ci + 2] = cellvl[ci];
            cellvl[ci] = v;
        }
    }
}

//=====================================================================
// function to compute the volume of a tetrahedra cell
//=====================================================================
float Fluent::tetraVol(float p0[3], float p1[3], float p2[3], float p3[3])
{
    //returns the volume of the tetrahedra cell
    float vol;

    vol = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return vol;
}

int
Fluent::readFile(const char *fileName)
{
    int zoneID;
    if (file.open(fileName) < 0)
    {
        sendError("ERROR: could not open file %s", fileName);
        return -1;
    }
    tetOnly = 1;
    numCellZones = 0;
    while (!file.eof() && (file.getCurrentSection() >= 0))
    {
        switch (file.getSection())
        {
        case -1:
            break;
        case 4: // Machine Config
        {

            int machine;
            file.nextSubSection();
            file.readDez(machine);
            file.readDez(bigEndian);
            file.skipSection(); //done
            if (!bigEndian)
                sendInfo("File is written byte swapped");
            else
                sendInfo("ByteOrder ist big endian!");
            /*
              fprintf(fd,"\n\
                       (0 \"Machine Config:\")\n\
                       (%d (%d %d %d %d %d %d %d %d %d %d %d))\n\
                       ",
                       XF_MACHINE_CONFIG,
                       my_machine_config.machine,
                       my_machine_config.big_endian,
                       my_machine_config.fp_format,
                       my_machine_config.sizeof_char,
                       my_machine_config.sizeof_short,
            my_machine_config.sizeof_int,
            my_machine_config.sizeof_long,
            my_machine_config.sizeof_float,
            my_machine_config.sizeof_double,
            my_machine_config.sizeof_real,
            my_machine_config.sizeof_pointer);*/
        }
        break;
        case 0: // comment
            file.readString(tmpBuf);
            sendInfo("%s", tmpBuf);
            file.skipSection();
            break;
        case 33: // GridSize
            file.nextSubSection();
            file.readDez(numCells);
            file.readDez(numFaces);
            file.readDez(numNodes);
            file.skipSection(); //done
            sendInfo("Grid Size: %d %d %d", numCells, numFaces, numNodes);
            break;
        case 2058:
        case 58: // Cell tree (hanging nodes meshes)
        {

            file.nextSubSection();
            int firstIndex;
            int lastIndex;
            int parent_zone_id;
            int child_zone_id;
            int i;
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readHex(parent_zone_id);
            file.readHex(child_zone_id);
            if (cellFlag)
            {
                for (i = firstIndex; i <= lastIndex; i++)
                {
                    cellFlag[i] = 1;
                }
            }
            //numCells -= ((lastIndex - firstIndex)+1);
            //sprintf(tmpBuf,"Ignore parent cells NumCells now: %d",numCells);
            //sendInfo(tmpBuf);
            file.skipSection(); //done
        }
        break;
        case 2061:
        case 61: // interface Face Parents (nonconformal interfaces)
        {

            file.nextSubSection();
            int firstIndex;
            int lastIndex;
            int i;
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            if (faceFlag)
            {
                for (i = firstIndex - 1; i < lastIndex; i++)
                {
                    faceFlag[i] = 1;
                }
            }
            file.skipSection(); //done
        }
        break;
        case 12: // Cells or Cell declaration
        {
            file.nextSubSection();
            file.readHex(zoneID);
            int firstIndex;
            int lastIndex;
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            if (zoneID == 0)
            {
                numCells = lastIndex;
                sendInfo("NumCells: %d", numCells);
                delete[] cellFlag;
                cellFlag = new char[numCells + 1];
                memset(cellFlag, 0, numCells + 1);
            }
            else
            {
                if (numCellZones >= MAX_CELL_ZONES)
                {
                    sendInfo("MAX_CELL_ZONES exceeded, contact developers!");
                    break;
                }
                cellZoneIds[numCellZones] = zoneID;
                numCellZones++;
            }
            file.skipSection(); //done
        }
        break;

        case 2012: // Cells or Cell declaration
        {
            file.nextSubSection();
            file.readHex(zoneID);
            int firstIndex;
            int lastIndex;
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            if (zoneID == 0)
            {
                numCells = lastIndex;
                sendInfo("NumCells: %d", numCells);
                delete[] cellFlag;
                cellFlag = new char[numCells + 1];
                memset(cellFlag, 0, numCells + 1);
            }
            else
            {
                if (numCellZones >= MAX_CELL_ZONES)
                {
                    sendInfo("MAX_CELL_ZONES exceeded, contact developers!");
                    break;
                }
                cellZoneIds[numCellZones] = zoneID;
                numCellZones++;
            }
            file.skipSection(); //done
        }
        break;

        case 3012: // Cells or Cell declaration
        {
            file.nextSubSection();
            file.readHex(zoneID);
            int firstIndex;
            int lastIndex;
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            if (zoneID == 0)
            {
                numCells = lastIndex;
                sendInfo("NumCells: %d", numCells);
                delete[] cellFlag;
                cellFlag = new char[numCells + 1];
                memset(cellFlag, 0, numCells + 1);
            }
            else
            {
                if (numCellZones >= MAX_CELL_ZONES)
                {
                    sendInfo("MAX_CELL_ZONES exceeded, contact developers!");
                    break;
                }
                cellZoneIds[numCellZones] = zoneID;
                numCellZones++;
            }
            file.skipSection(); //done
        }
        break;

        // ASCII data
        case 10: // Nodes or node declaration
        {
            int firstIndex;
            int lastIndex, type, i, ND;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readDez(type);

            file.readDez(ND); // optional
            if (ND < 0)
                ND = 3; // If not exists, assume 3D

            if (zoneID == 0)
            {
                numNodes = lastIndex;
                sendInfo("NumNodes: %d", numNodes);
                delete[] x_coords;
                delete[] y_coords;
                delete[] z_coords;
                x_coords = new float[numNodes];
                y_coords = new float[numNodes];
                z_coords = new float[numNodes];
            }
            else
            {
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();
                    for (i = firstIndex - 1; i < lastIndex; i++)
                    {
                        file.readFloat(x_coords[i]);
                        file.readFloat(y_coords[i]);
                        if (ND > 2)
                            file.readFloat(z_coords[i]);
                        else
                            z_coords[i] = 0.0f;
                    }
                }
            }
            file.skipSection(); //done
        }
        break;
        // binary single
        case 2010: // Binary Nodes or node declaration
        {
            int firstIndex;
            int lastIndex, type, i, ND;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readDez(type);

            file.readDez(ND); // optional
            if (ND < 0)
                ND = 3; // If not exists, assume 3D

            if (zoneID == 0)
            {
                numNodes = lastIndex;
                sendInfo("NumNodes (binary single): %d", numNodes);
                delete[] x_coords;
                delete[] y_coords;
                delete[] z_coords;
                x_coords = new float[numNodes];
                y_coords = new float[numNodes];
                z_coords = new float[numNodes];
            }
            else
            {
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();
                    for (i = firstIndex - 1; i < lastIndex; i++)
                    {
                        file.readBin(x_coords[i]);
                        file.readBin(y_coords[i]);
                        if (ND > 2)
                            file.readBin(z_coords[i]);
                        else
                            z_coords[i] = 0.0f;
                        //                       cerr << "Fluent::readFile info: x[" << i << "] = " << x_coords[i]
                        //                             << ", y[" << i << "] = " << y_coords[i]
                        //                            << ", z[" << i << "] = " << z_coords[i] << endl;
                    }
                }
            }
            file.skipSection(); //done
        }
        break;
        // binary double
        case 3010: // Binary Nodes or node declaration
        {
            int firstIndex;
            int lastIndex, type, i, ND;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readDez(type);

            file.readDez(ND); // optional
            if (ND < 0)
                ND = 3; // If not exists, assume 3D

            if (zoneID == 0)
            {
                numNodes = lastIndex;
                sendInfo("NumNodes (binary double): %d", numNodes);
                delete[] x_coords;
                delete[] y_coords;
                delete[] z_coords;

                x_coords = new float[numNodes];
                y_coords = new float[numNodes];
                z_coords = new float[numNodes];
            }
            else
            {
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();
                    double xt, yt, zt;
                    for (i = firstIndex - 1; i < lastIndex; i++)
                    {
                        file.readBin(xt);
                        x_coords[i] = (float)xt;

                        file.readBin(yt);
                        y_coords[i] = (float)yt;

                        if (ND > 2)
                        {
                            file.readBin(zt);
                            z_coords[i] = (float)zt;
                        }
                        else
                        {
                            z_coords[i] = 0.0f;
                        }
                    }
                }
            }
            file.skipSection(); //done
        }
        break;

        case 13: // Faces or Faces declaration
        {
            int firstIndex;
            int lastIndex, type, globalElementType, elementType, n, i;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readHex(type);
            file.readDez(globalElementType);

            // 				    cerr << "Zone "
            // 					 << zoneID << " fi " << firstIndex << " li " << lastIndex
            // 					 << " t " << type << " et " << globalElementType << endl;

            if (zoneID == 0)
            {
                numFaces = lastIndex;
                sendInfo("numFaces: %d", numFaces);
                delete[] vertices;
                delete[] typelist;
                delete[] facetype;
                delete[] faceFlag;
                delete[] leftNeighbor;
                delete[] rightNeighbor;
                vertices = new int[numFaces * 4];
                typelist = new int[numFaces];
                facetype = new int[numFaces];
                faceFlag = new char[numFaces];
                leftNeighbor = new int[numFaces];
                rightNeighbor = new int[numFaces];
                memset(faceFlag, 0, numFaces);
                int ii;
                for (ii = 0; ii < numFaces; ++ii)
                {
                    leftNeighbor[ii] = 0;
                    rightNeighbor[ii] = 0;
                }
            }
            else
            {
                if (globalElementType != 3)
                    tetOnly = 0;
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();

                    if (globalElementType != 0)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            typelist[i] = globalElementType;
                            facetype[i] = type;
                            if (type > 1000)
                            {
                                faceFlag[i] = 1; // this is a face from nonconformal
                            }
                            for (n = 0; n < globalElementType; n++)
                                file.readHex(vertices[i * 4 + n]);
                            //neighborCells
                            file.readHex(leftNeighbor[i]);
                            file.readHex(rightNeighbor[i]);
                        }
                    }
                    else
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            file.readDez(elementType);
                            typelist[i] = elementType;
                            facetype[i] = type;
                            if (type > 1000)
                            {
                                faceFlag[i] = 1; // this is a face from nonconformal
                            }
                            for (n = 0; n < elementType; n++)
                            {
                                file.readHex(vertices[i * 4 + n]);
                            }
                            //neighborCells
                            file.readHex(leftNeighbor[i]);
                            file.readHex(rightNeighbor[i]);
                        }
                    }
                }
            }
            file.skipSection(); //done
        }
        break;

        case 2013: // Binary Faces or Faces declaration
        {
            int firstIndex;
            int lastIndex, type, globalElementType, elementType, n, i;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readHex(type);
            file.readDez(globalElementType);

            // 	    cerr << "Zone "
            // 		 << zoneID
            // 		 << " fi "
            // 		 << firstIndex
            // 		 << " li "
            // 	     				   << lastIndex
            // 		 << " no of nodes: "
            // 		 << lastIndex - firstIndex
            // 		 <<   " t "
            // 		 << type
            // 		 << " et "
            // 		 << globalElementType << endl;

            if (zoneID == 0)
            {
                numFaces = lastIndex;
                sendInfo("numFaces: %d", numFaces);
                delete[] vertices;
                delete[] typelist;
                delete[] facetype;
                delete[] leftNeighbor;
                delete[] rightNeighbor;
                vertices = new int[numFaces * 4];
                typelist = new int[numFaces];
                facetype = new int[numFaces];
                leftNeighbor = new int[numFaces];
                rightNeighbor = new int[numFaces];
                int ii;
                for (ii = 0; ii < numFaces; ++ii)
                {
                    leftNeighbor[ii] = 0;
                    rightNeighbor[ii] = 0;
                }
            }
            else
            {
                if (globalElementType != 3)
                    tetOnly = 0;
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();
                    if (globalElementType != 0)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            typelist[i] = globalElementType;
                            facetype[i] = type;
                            for (n = 0; n < globalElementType; n++)
                            {
                                file.readBin(vertices[i * 4 + n]);
                            }
                            int idat;
                            file.readBin(idat); //neighborCells
                            leftNeighbor[i] = idat;
                            // cerr << "Application::readFile(..) -1-  sect 2013 " << i << " leftNeighbor " << idat << endl;
                            file.readBin(rightNeighbor[i]);
                        }
                    }
                    else
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            file.readBin(elementType);
                            typelist[i] = elementType;
                            facetype[i] = type;
                            for (n = 0; n < elementType; n++)
                            {
                                file.readBin(vertices[i * 4 + n]);
                            }
                            int idat;
                            file.readBin(idat); //neighborCells
                            //			    cerr << "Application::readFile(..) -2- sect 2013 " << i << " leftNeighbor " << idat << endl;
                            leftNeighbor[i] = idat;
                            file.readBin(rightNeighbor[i]);
                        }
                    }
                }
            }

            file.skipSection(); //done
        }
        break;

        // binary double
        case 3013: // Binary Faces or Faces declaration
        {
            int firstIndex;
            int lastIndex, type, globalElementType, elementType, n, i;
            file.nextSubSection();
            file.readHex(zoneID);
            file.readHex(firstIndex);
            file.readHex(lastIndex);
            file.readHex(type);
            file.readDez(globalElementType);

            // 			      cerr << "Zone "
            // 				   << zoneID
            // 				   << " fi "
            // 				   << firstIndex
            // 				   << " li "
            // 				   << lastIndex
            // 				   << " no of nodes: "
            // 				   << lastIndex - firstIndex
            // 				   <<   " t "
            // 				   << type
            // 				   << " et "
            // 				   << globalElementType << endl;

            if (zoneID == 0)
            {
                numFaces = lastIndex;
                sendInfo("numFaces: %d", numFaces);
                delete[] vertices;
                delete[] typelist;
                delete[] facetype;
                delete[] leftNeighbor;
                delete[] rightNeighbor;
                vertices = new int[numFaces * 4];
                typelist = new int[numFaces];
                facetype = new int[numFaces];
                leftNeighbor = new int[numFaces];
                rightNeighbor = new int[numFaces];
                int ii;
                for (ii = 0; ii < numFaces; ++ii)
                {
                    leftNeighbor[ii] = 0;
                    rightNeighbor[ii] = 0;
                }
            }
            else
            {
                if (globalElementType != 3)
                    tetOnly = 0;
                if (numNodes > 0)
                {
                    file.endSubSection();
                    file.nextSubSection();
                    if (globalElementType != 0)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            typelist[i] = globalElementType;
                            facetype[i] = type;
                            for (n = 0; n < globalElementType; n++)
                            {
                                file.readBin(vertices[i * 4 + n]);
                            }
                            //neighborCells
                            file.readBin(leftNeighbor[i]);
                            file.readBin(rightNeighbor[i]);
                        }
                    }
                    else
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            file.readBin(elementType);
                            typelist[i] = elementType;
                            facetype[i] = type;
                            for (n = 0; n < elementType; n++)
                            {
                                file.readBin(vertices[i * 4 + n]);
                            }
                            //neighborCells
                            file.readBin(leftNeighbor[i]);
                            file.readBin(rightNeighbor[i]);
                        }
                    }
                }
            }
            file.skipSection(); //done
            break;
        }

        default:
            file.skipSection();
            break;
        }
    }

    file.close();
    return 0;
}

void Fluent::updateChoice()
{
    char *old[3];
    int oldChoice[3];

    for (int i = 0; i < 3; i++)
    {
        oldChoice[i] = p_data[i]->getValue();

        if (p_data[i]->getActLabel())
        {
            old[i] = new char[strlen(p_data[i]->getActLabel()) + 1];
            strcpy(old[i], p_data[i]->getActLabel());
        }
        else
            old[i] = NULL;
    }

    const char **choices = new const char *[numVars + 2];
    choices[0] = "(none)";
    for (int i = 0; i < numVars; i++)
    {
        if (varIsFace[i])
        {
            if (varTypes[i] < 0)
                choices[i + 1] = FluentVecFaceVarNames[-(varTypes[i])];
            else
                choices[i + 1] = FluentFaceVarNames[varTypes[i]];
        }
        else
        {
            if (varTypes[i] < 0)
                choices[i + 1] = FluentVecVarNames[-(varTypes[i])];
            else
                choices[i + 1] = FluentVarNames[varTypes[i]];
        }

        for (int j = 0; j < 3; j++)
        {
            if (old[j] && !strcmp(choices[i + 1], old[j]))
                oldChoice[j] = i + 1;
        }
    }
    choices[numVars + 1] = NULL;
    for (int i = 0; i < 3; i++)
    {
        p_data[i]->setValue(numVars + 1, (char **)choices, oldChoice[i]);
        delete[] old[i];
    }
    delete[] choices;
}

void Fluent::addVariable(int varNum)
{
    int i;
    if (varNum > 309)
    {
        sendInfo("Variables %d (> 309) not supported", varNum);
        return;
    }
    for (i = 0; i < numVars; i++)
    {
        if ((varTypes[i] == varNum) && (varIsFace[i] == 0))
        {
            return;
        }
    }
    sendInfo("Variable %s (%d)", FluentVarNames[varNum], varNum);
    varTypes[numVars] = varNum;
    varIsFace[numVars] = 0;
    numVars++;
    for (i = 1; i < NumVectVars; i++)
    {
        if (varNum == VectorScalarMap[i]) // this is a vector
        {
            varTypes[numVars] = -i;
            varIsFace[numVars] = 0;
            numVars++;
        }
    }
}

void Fluent::addFaceVariable(int varNum)
{

    int i;
    if (varNum > 150)
    {
        sendInfo("Variables %d (> 150) not supported", varNum);
        return;
    }
    for (i = 0; i < numVars; i++)
    {
        if ((varTypes[i] == varNum) && (varIsFace[i] == 1))
        {
            return;
        }
    }
    sendInfo("Variable %s (%d)", FluentFaceVarNames[varNum], varNum);
    varTypes[numVars] = varNum;
    varIsFace[numVars] = 1;
    numVars++;
    for (i = 1; i < NumVectVars; i++)
    {
        if (varNum == VectorScalarMap[i]) // this is a vector
        {
            varTypes[numVars] = -i;
            varIsFace[numVars] = 1;
            numVars++;
        }
    }
}

int Fluent::parseDat(const char *fileName)
{
    int zoneID, dataType, i;

    numVars = 0;
    if (file.open(fileName) < 0)
    {
        sendInfo("ERROR: could not open file %s", fileName);
        return -1;
    }
    while (!file.eof() && (file.getCurrentSection() >= 0))
    {
        switch (file.getSection())
        {
        case -1:
            break;
        case 0: // comment
            file.readString(tmpBuf);
            sendInfo("%s", tmpBuf);
            file.skipSection();
            break;
        case 33: // GridSize
            file.nextSubSection();
            file.readDez(numCells);
            int datNumFaces;
            file.readDez(datNumFaces);
            file.readDez(numNodes);
            file.skipSection(); //done
            sendInfo("Data Grid Size: %d %d %d", numCells, datNumFaces, numNodes);
            break;
        case 300: // DataBlock
        case 2300: // BinaryDataBlock
        {
            int cellVar = 0;
            file.nextSubSection();
            file.readDez(dataType);
            file.readDez(zoneID);
            for (i = 0; i < numCellZones; i++)
            {
                if (zoneID == cellZoneIds[i])
                {
                    addVariable(dataType);
                    cellVar = 1;
                    break;
                }
            }
            if (!cellVar)
            {
                addFaceVariable(dataType);
            }
            file.skipSection(); //done
        }
        break;

        case 3300: // BinaryDataBlock
        {
            int cellVar = 0;
            file.nextSubSection();
            file.readDez(dataType);
            file.readDez(zoneID);
            for (i = 0; i < numCellZones; i++)
            {
                if (zoneID == cellZoneIds[i])
                {
                    addVariable(dataType);
                    cellVar = 1;
                    break;
                }
            }
            if (!cellVar)
            {
                addFaceVariable(dataType);
            }
            file.skipSection(); //done
        }
        break;
        default:
            sendInfo("Skipping Section %d", file.getCurrentSection());
            file.skipSection();
            break;
        }
    }

    file.close();
    return 0;
}

int
Fluent::readDat(const char *fileName, float *dest, int var, int type, float *dest1, float *dest2)
{
    int zoneID, dataType;
    numVars = 0;
    if (file.open(fileName) < 0)
    {
        sendError("ERROR: could not open file %s", fileName);
        return -1;
    }
    while (!file.eof() && (file.getCurrentSection() >= 0))
    {
        switch (file.getSection())
        {
        case -1:
            break;
        case 0: // comment
            file.readString(tmpBuf);
            sendInfo("%s", tmpBuf);
            file.skipSection();
            break;
        case 33: // GridSize
            file.nextSubSection();
            file.readDez(numCells);
            int datNumFaces;
            file.readDez(datNumFaces);
            file.readDez(numNodes);
            file.skipSection(); //done
            //sendInfo("Grid Size: %d %d %d",numCells,numFaces,numNodes);
            break;
        case 2300: // BinaryDataBlock
        {
            int size, dummy, i, firstIndex, lastIndex, isCell = 0;
            float dat;
            file.nextSubSection();
            file.readDez(dataType);
            file.readDez(zoneID);
            file.readDez(size);
            file.readDez(dummy);
            file.readDez(dummy);
            file.readDez(firstIndex);
            file.readDez(lastIndex);
            if (dataType == var)
            {
                file.endSubSection();
                file.nextSubSection();
                if (type == READ_CELLS)
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                        {
                            for (i = firstIndex - 1; i < lastIndex; i++)
                            {
                                int idx = elemToCells_[i];
                                float dummy;
                                if (idx > -1)
                                {
                                    file.readBin(dest[idx]);
                                    if (size == 3)
                                    {
                                        if (dest1)
                                            file.readBin(dest1[idx]);
                                        if (dest2)
                                            file.readBin(dest2[idx]);
                                    }
                                }
                                else
                                {
                                    file.readBin(dummy);
                                    if (size == 3)
                                    {
                                        file.readBin(dummy);
                                        file.readBin(dummy);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                else
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                            isCell = 1;
                    }
                    if (!isCell)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            int idx = elemToCells_[i];
                            float dummy;
                            if (idx > -1)
                            {
                                int thisType = typelist[i];
                                if (thisType <= 4)
                                {
                                    file.readBin(dat);
                                    dest[vertices[i * 4 + 0] - 1] = dat;
                                    dest[vertices[i * 4 + 1] - 1] = dat;
                                    dest[vertices[i * 4 + 2] - 1] = dat;
                                    if (thisType == 4)
                                        dest[vertices[i * 4 + 3] - 1] = dat;

                                    if (size == 3)
                                    {
                                        file.readBin(dat);
                                        if (dest1)
                                            dest1[vertices[i * 4 + 0] - 1] = dat;
                                        if (dest1)
                                            dest1[vertices[i * 4 + 1] - 1] = dat;
                                        if (dest1)
                                            dest1[vertices[i * 4 + 2] - 1] = dat;
                                        if (thisType == 4)
                                        {
                                            if (dest1)
                                                dest1[vertices[i * 4 + 3] - 1] = dat;
                                        }

                                        file.readBin(dat);
                                        if (dest2)
                                            dest2[vertices[i * 4 + 0] - 1] = dat;
                                        if (dest2)
                                            dest2[vertices[i * 4 + 1] - 1] = dat;
                                        if (dest2)
                                            dest2[vertices[i * 4 + 2] - 1] = dat;
                                        if (thisType == 4)
                                        {
                                            if (dest2)
                                                dest2[vertices[i * 4 + 3] - 1] = dat;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                file.readBin(dummy);
                            }
                            if (size == 3)
                            {
                                file.readBin(dummy);
                                file.readBin(dummy);
                            }
                        }
                    }
                }
            }
            file.skipSection(); //done
        }
        break;

        case 3300: // BinaryDataBlock
        {
            int size, dummy, i, firstIndex, lastIndex, isCell = 0;
            double dat;
            file.nextSubSection();
            file.readDez(dataType);
            file.readDez(zoneID);
            file.readDez(size);
            file.readDez(dummy);
            file.readDez(dummy);
            file.readDez(firstIndex);
            file.readDez(lastIndex);
            if (dataType == var)
            {
                file.endSubSection();
                file.nextSubSection();
                if (type == READ_CELLS)
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                        {
                            for (i = firstIndex - 1; i < lastIndex; i++)
                            {
                                int idx = elemToCells_[i];
                                double dummy;
                                if (idx > -1)
                                {
                                    file.readBin(dat);
                                    dest[idx] = (float)dat;
                                    if (size == 3)
                                    {
                                        if (dest1)
                                        {
                                            file.readBin(dat);
                                            dest1[idx] = (float)dat;
                                        }
                                        if (dest2)
                                        {
                                            file.readBin(dat);
                                            dest2[idx] = (float)dat;
                                        }
                                    }
                                }
                                else
                                {
                                    file.readBin(dummy);
                                    if (size == 3)
                                    {
                                        file.readBin(dummy);
                                        file.readBin(dummy);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                else
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                        {
                            isCell = 1;
                        }
                    }
                    if (!isCell)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            int idx = elemToCells_[i];
                            double dummy;
                            if (idx > -1)
                            {
                                int thisType = typelist[i];
                                if (thisType <= 4)
                                {
                                    file.readBin(dat);
                                    dest[vertices[i * 4 + 0] - 1] = (float)dat;
                                    dest[vertices[i * 4 + 1] - 1] = (float)dat;
                                    dest[vertices[i * 4 + 2] - 1] = (float)dat;
                                    if (thisType == 4)
                                        dest[vertices[i * 4 + 3] - 1] = (float)dat;

                                    if (size == 3)
                                    {
                                        file.readBin(dat);
                                        if (dest1)
                                            dest1[vertices[i * 4 + 0] - 1] = (float)dat;
                                        if (dest1)
                                            dest1[vertices[i * 4 + 1] - 1] = (float)dat;
                                        if (dest1)
                                            dest1[vertices[i * 4 + 2] - 1] = (float)dat;
                                        if (thisType == 4)
                                            if (dest1)
                                                dest1[vertices[i * 4 + 3] - 1] = (float)dat;

                                        file.readBin(dat);
                                        if (dest2)
                                            dest2[vertices[i * 4 + 0] - 1] = (float)dat;
                                        if (dest2)
                                            dest2[vertices[i * 4 + 1] - 1] = (float)dat;
                                        if (dest2)
                                            dest2[vertices[i * 4 + 2] - 1] = (float)dat;
                                        if (thisType == 4)
                                            if (dest2)
                                                dest2[vertices[i * 4 + 3] - 1] = (float)dat;
                                    }
                                }
                            }
                            else
                            {
                                file.readBin(dummy);
                            }
                            if (size == 3)
                            {
                                file.readBin(dummy);
                                file.readBin(dummy);
                            }
                        }
                    }
                }
            }
            file.skipSection(); //done
        }
        break;

        case 300: // DataBlock
        {
            int size, dummy, i, firstIndex, lastIndex, isCell = 0;
            float dat;
            file.nextSubSection();
            file.readDez(dataType);
            file.readDez(zoneID);
            file.readDez(size);
            file.readDez(dummy);
            file.readDez(dummy);
            file.readDez(firstIndex);
            file.readDez(lastIndex);
            if (dataType == var)
            {
                file.endSubSection();
                file.nextSubSection();

                if (type == READ_CELLS)
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                        {
                            for (i = firstIndex - 1; i < lastIndex; i++)
                            {
                                int idx = elemToCells_[i];
                                float dummy;
                                if (idx > -1)
                                {
                                    file.readFloat(dest[idx]);
                                    if (size == 3)
                                    {
                                        if (dest1)
                                            file.readFloat(dest1[idx]);
                                        if (dest2)
                                            file.readFloat(dest2[idx]);
                                    }
                                }
                                else
                                {
                                    file.readFloat(dummy);
                                    if (size == 3)
                                    {
                                        file.readFloat(dummy);
                                        file.readFloat(dummy);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                else
                {
                    for (i = 0; i < numCellZones; i++)
                    {
                        if (zoneID == cellZoneIds[i])
                        {
                            isCell = 1;
                        }
                    }
                    if (!isCell)
                    {
                        for (i = firstIndex - 1; i < lastIndex; i++)
                        {
                            int idx = elemToCells_[i];
                            float dummy;
                            if (idx > -1)
                            {
                                file.readFloat(dat);
                                dest[vertices[i * 4 + 0] - 1] = dat;
                                dest[vertices[i * 4 + 1] - 1] = dat;
                                dest[vertices[i * 4 + 2] - 1] = dat;
                                if (typelist[i] == 4)
                                    dest[vertices[i * 4 + 3] - 1] = dat;

                                if (size == 3)
                                {
                                    file.readFloat(dat);
                                    if (dest1)
                                        dest1[vertices[i * 4 + 0] - 1] = dat;
                                    if (dest1)
                                        dest1[vertices[i * 4 + 1] - 1] = dat;
                                    if (dest1)
                                        dest1[vertices[i * 4 + 2] - 1] = dat;
                                    if (typelist[i] == 4)
                                        if (dest1)
                                            dest1[vertices[i * 4 + 3] - 1] = dat;

                                    file.readFloat(dat);
                                    if (dest2)
                                        dest2[vertices[i * 4 + 0] - 1] = dat;
                                    if (dest2)
                                        dest2[vertices[i * 4 + 1] - 1] = dat;
                                    if (dest2)
                                        dest2[vertices[i * 4 + 2] - 1] = dat;
                                    if (typelist[i] == 4)
                                        if (dest2)
                                            dest2[vertices[i * 4 + 3] - 1] = dat;
                                }
                            }
                            else
                            {
                                file.readFloat(dummy);
                                if (size == 3)
                                {
                                    file.readFloat(dummy);
                                    file.readFloat(dummy);
                                }
                            }
                        }
                    }
                }
            }
            file.skipSection(); //done
        }
        break;
        default:
            file.skipSection();
            break;
        }
    }

    file.close();
    return 0;
}

fluentFile::fluentFile()
{
    currentChar = buf;
    lastChar = buf - 1;
    numback = 0;

    // get runtime number-representation
    int num(0);
    char *t = (char *)&num;

    t[0] = (char)1;
    t[1] = '\0';
    t[2] = '\0';
    t[3] = '\0';

    if (num == 1)
    {
        myByteOrder_ = fluentFile::LITTLE_END;
    }
    else if (num == 16777216)
    {
        myByteOrder_ = fluentFile::BIG_END;
    }
    else
    {
        cerr << "fluentFile::fluentFile(): unknown byte-order" << endl;
    }
}

fluentFile::~fluentFile()
{
    currentSection = 0;
}

int fluentFile::open(const char *fileName)
{
#ifdef _WIN32
    _fmode = _O_BINARY;
    fd = Covise::open((char *)fileName, _O_RDONLY | _O_BINARY);
    // Set "stdin" to have binary mode:
    if (fd > 0)
    {
        int result = _setmode(fd, _O_BINARY);
        if (result == -1)
            perror("Cannot set mode");
        else
            printf("'fluentFile' successfully changed to binary mode\n");
    }

#else
    fd = Covise::open((char *)fileName, O_RDONLY);
#endif
    numback = 0;
    currentChar = buf;
    currentSection = 0;
    lastChar = buf - 1;
    if (fd > 0)
    {
        fillBuf();
    }
    return fd;
}

void fluentFile::close()
{
    if (fd > 0)
        ::close(fd);

    currentChar = buf;
    lastChar = buf - 1;
}

int fluentFile::fillBuf()
{
    int numRead = 0, i, numLeft;
    if (lastChar <= buf)
    {
        numRead = read(fd, buf, BUFSIZE);
        if (numRead <= 0)
        {
            currentChar = buf;
            lastChar = buf - 1;
            return -1;
        }
        currentChar = buf;
        lastChar = buf + numRead - 1;
        return numRead;
    }
    numLeft = (lastChar - currentChar) + 1;
    for (i = 0; i < numLeft; i++)
    {
        buf[i] = currentChar[i];
    }
    numRead = read(fd, buf + numLeft, BUFSIZE - numLeft);
    if (numRead < 0)
    {
        currentChar = buf;
        lastChar = buf - 1;
        return -1;
    }
    lastChar = buf + numLeft + numRead - 1;
    currentChar = buf + numLeft;
    if (eof())
        return -1;
    return numLeft + numRead;
}

int fluentFile::getSection()
{
    char c;
    numOpen = 0;
    while (!eof())
    {
        c = getChar();
        if (c == '(')
        {
            if (readDez(currentSection) < 0)
                return -1;
            else
            {
                numOpen = 1;
                gotHeader = 1;
                if (currentSection > 2000)
                {
                    gotHeader = 0;
                }

                return currentSection;
            }
        }
    }
    return -1;
}

int fluentFile::nextSubSection()
{
    char c;
    while (!eof())
    {
        c = getChar();
        if (c == '(')
        {
            numOpen++;
            if (numOpen == 2)
                gotHeader = 1;
            return 0;
        }
        if (c == ')')
        {
            numOpen--;
        }
    }
    return -1;
}

int fluentFile::endSubSection()
{
    char c;
    while (!eof())
    {
        c = getChar();
        if (c == ')')
        {
            gotHeader = 1;
            numOpen--;
            return 0;
        }
    }
    return -1;
}

int fluentFile::skipSection()
{
    char c;
    int i;
    const char *endMarker = "End of Binary Section";
    //if(currentSection>2000)
    //{
    //    gotHeader = 0;
    //}
    while (numOpen)
    {
        c = getChar();

        if (c == '(')
        {
            if ((currentSection < 2000) || (gotHeader == 0))
            {
                numOpen++;
            }
        }
        if (c == ')')
        {
            if (currentSection > 2000)
            {
                if (gotHeader == 0)
                {
                    numOpen--;
                    if (numOpen == 1)
                    {
                        gotHeader = 1;
                    }
                }
            }
            else
                numOpen--;
        }
        if ((currentSection > 2000) && (gotHeader))
        {
            for (i = 0; i < 21; i++)
            {
                if (endMarker[i] != c)
                    break;
                c = getChar();
            }
            if (i == 21)
            {
                currentSection = 0;
                numOpen = 1;
            }
        }
        if (eof())
            return -1;
    }
    return 0;
}

int fluentFile::readDez(int &num)
{
    if (readString(tmpBuf) < 0)
        return -1;
    if (sscanf(tmpBuf, "%d", &num) < 1)
        return -1;
    return 0;
}

int fluentFile::readHex(int &num)
{
    if (readString(tmpBuf) < 0)
        return -1;
    if (sscanf(tmpBuf, "%x", &num) < 1)
        return -1;
    return 0;
}

int fluentFile::readFloat(float &num)
{
    if (readString(tmpBuf) < 0)
        return -1;
    if (sscanf(tmpBuf, "%f", &num) < 1)
        return -1;
    return 0;
}

void fluentFile::readBin(float &num)
{
    char *buf = (char *)&num;

    bool swap = ((bigEndian ^ (myByteOrder_ == fluentFile::BIG_END)) != 0);

    if (swap)
    {
        buf[3] = getChar();
        buf[2] = getChar();
        buf[1] = getChar();
        buf[0] = getChar();
    }
    else
    {
        buf[0] = getChar();
        buf[1] = getChar();
        buf[2] = getChar();
        buf[3] = getChar();
    }
}

void
fluentFile::readBin(double &num)
{
    char *buf = (char *)&num;

    bool swap = ((bigEndian ^ (myByteOrder_ == fluentFile::BIG_END)) != 0);

    if (swap)
    {
        buf[7] = getChar();
        buf[6] = getChar();
        buf[5] = getChar();
        buf[4] = getChar();
        buf[3] = getChar();
        buf[2] = getChar();
        buf[1] = getChar();
        buf[0] = getChar();
    }
    else
    {
        buf[0] = getChar();
        buf[1] = getChar();
        buf[2] = getChar();
        buf[3] = getChar();
        buf[4] = getChar();
        buf[5] = getChar();
        buf[6] = getChar();
        buf[7] = getChar();
    }
}

void
fluentFile::readBin(int &num)
{
    char *buf = (char *)&num;

    bool swap = ((bigEndian ^ (myByteOrder_ == fluentFile::BIG_END)) != 0);

    if (swap)
    {
        buf[3] = getChar();
        buf[2] = getChar();
        buf[1] = getChar();
        buf[0] = getChar();
    }
    else
    {
        buf[0] = getChar();
        buf[1] = getChar();
        buf[2] = getChar();
        buf[3] = getChar();
    }
}

int fluentFile::readString(char *str)
{
    char *s = str, c = ' ';

    while (!eof() && ((c == ' ') || (c == '\t') || (c == '\r') || (c == '\n')))
    {
        c = getChar();
    }
    if ((c == ')') || (c == '('))
    {
        putBack(c);
        *s = '\0';
        return 0;
    }
    if (c == '"')
    {
        while (!eof())
        {
            c = getChar();
            if (c == '"')
            {
                *s = '\0';
                return 0;
            }
            *s = c;
            s++;
        }
        *s = '\0';
        return -1;
    }
    *s = c;
    s++;
    while (!eof())
    {
        c = getChar();
        if ((c == ' ') || (c == ')') || (c == '(') || (c == '\t') || (c == '\r') || (c == '\n'))
        {
            putBack(c);
            *s = '\0';
            return 0;
        }
        *s = c;
        s++;
    }
    *s = '\0';
    return -1;
}

coDoLines *Fluent::makeLines()
{

    cerr << "Fluent::makeLines info: making lines for " << numFaces << " faces" << endl;

    int *visited = new int[numFaces];

    int *lines = new int[numFaces];
    int *corners = new int[numFaces * 2];

    memset(visited, 0, numFaces * sizeof(int));

    memset(lines, 0, numFaces * sizeof(int));
    memset(corners, 0, numFaces * 2 * sizeof(int));

    int numLines = 0;
    int numCorners = 0;

    for (int ctr = 0; ctr < numFaces - 1; ++ctr)
    {

        for (; ctr < numFaces - 1; ++ctr)
        {

            if (visited[ctr] || typelist[ctr] != 2)
            {
                //cerr << "Fluent::makeLines info: skipping face " << ctr << endl;
                break;
            }
            else
            {

                //cerr << "Fluent::makeLines info: starting line " << numLines << endl;
                lines[numLines] = numCorners;

                //for (int i = ctr; ((i != 0) && (!visited[i])); i = rightNeighbor[i]) {
                //cerr << "Fluent::makeLines info: processing face " << i << endl;
                //cerr << "Fluent::makeLines info: vertex data: "
                //     << vertices[i*4] << ", " << vertices[i*4 + 1] << ", " << endl;
                //cerr << "Fluent::makeLines info: leftNeighbour = " << faces[leftNeighbor[i]] << endl;
                //cerr << "Fluent::makeLines info: rightNeighbour = " << faces[rightNeighbor[i]] << endl;

                corners[numCorners++] = vertices[4 * ctr];
                corners[numCorners++] = vertices[4 * ctr + 1];

                ++visited[ctr];

                if (leftNeighbor[ctr] == 0)
                    break;
            }

            //cerr << "Fluent::makeLines info: ending line " << numLines << endl;
            ++numLines;
        }
    }

    char *buf = new char[strlen(p_linePort->getObjName()) + 1];
    strcpy(buf, p_linePort->getObjName());

    coDoLines *rv = new coDoLines(buf, numNodes, x_coords, y_coords, z_coords, numCorners, corners, numLines, lines);

    delete[] buf;
    delete[] lines;
    delete[] corners;
    delete[] visited;

    return rv;
}

MODULE_MAIN(IO, Fluent)
