/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2001    **
 ** Author:                                                              **
 **                            Karin Mller                              **
 **                                         Vircinity                    **
 **                            Technologiezentrum                        **
 **                            70550 Stuttgart                           **
 ** Date:  01.10.01                                                      **
 **             Adaption to GLView 7.1 Thomas Speiser, Andreas Funke     **
\**************************************************************************/

#include "ReadVTF.h"
#include "Binary.h"
#include <alg/coCellToVert.h>
#include <util/coviseCompat.h>
#include <do/coDoSet.h>

#define DEBUG

//==============================================================================

ReadVTF::ReadVTF(int argc, char *argv[])
    : coModule(argc, argv, "Read VTF format")
{
    char name[10];
    int i = 0;

    ptBrowserParam = addFileBrowserParam("Filename", "path to the vtf file");
    ptBrowserParam->setValue("~/covise/data/testdata/vtf/", "*.vtf");

    p_Selection = addStringParam("Selection", "Number selection for parts");

    pt_outPort = addOutputPort("Grid", "UnstructuredGrid", "grid");
    pt_outPoly = addOutputPort("Polygons", "Polygons", "poly");

    p_Selection->setValue("0-9999999");
    // out port for scalar or vector data
    for (i = 0; i < NUMPORTS; i++)
    {
        sprintf(name, "dataout%d", i + 1);
        pt_outResults[i] = addOutputPort(name, "Float|Vec3", "S3D or V3D results");
    }

    // choice parameter
    for (i = 0; i < NUMPORTS; i++)
    {
        sprintf(name, "port_%d", i + 2);
        pt_choice[i] = addChoiceParam(name, "select typ of data");
        const char *noneChoice[] = { "---" };
        pt_choice[i]->setValue(1, noneChoice, 0);
    }
    tReadData = NULL;
    poly = NULL;
}

//==============================================================================

ReadVTF::~ReadVTF()
{
}

//==============================================================================

void ReadVTF::param(const char *paramname, bool inMapLoading)
{
    if (strcmp(ptBrowserParam->getName(), paramname) == 0)
    {
        m_cPath = ptBrowserParam->getValue();
        if (tReadData != NULL)
        {
            delete tReadData;
        }
        tReadData = new ReadVTFData(openFile(m_cPath));

        tReadData->setValues();
        tReadData->m_bResults = false;
        //m_cPath = ptBrowserParam->getValue();

        // get information of file an set choice parameter values
        getInfoAboutFile();

        if (tReadData->m_bResults)
        {
            if (!inMapLoading)
            {
                for (int i = 0; i < NUMPORTS; i++)
                {
                    pt_choice[i]->setValue(tReadData->m_iNumDes,
                                           tReadData->m_cDescription, 0);
                }
            }
        }
        else
        {
            for (int i = 0; i < NUMPORTS; i++)
            {
                const char *ch[] = { "---" };
                pt_choice[i]->setValue(1, ch, 0);
            }
        }
    }
}

//==============================================================================

int ReadVTF::compute(const char *)
{
    // initialize all values
    int iError = 1;

    // Get the selection
    const char *selection;
    // Covise::get_string_param("Selection",&selection);
    sel.clear();
    selection = p_Selection->getValue();
    sel.add(selection);

    m_cPath = ptBrowserParam->getValue();

    if (tReadData == NULL)
    {
        tReadData = new ReadVTFData(openFile(m_cPath));
    }

    tReadData->setValues();

    //allocate space for NODES, ELEMENT and RESULTS
    tReadData->m_tElem = new Element[tReadData->m_iNumElem];
    tReadData->m_tNodes = new Node[tReadData->m_iNumNodes];
    tReadData->m_tRes = new Results[tReadData->m_iNumRes];

    m_cPath = ptBrowserParam->getValue();

    // read all data and fill all structs in ReadVTFData
    iError = readFile();
    if (iError)
        return STOP_PIPELINE;

    // now draw everything
    if (tReadData->getVSCounter() > 0)
    {
        m_iNumTimeSteps = tReadData->m_tViewS[0].header.iNumSteps;
    }
    else if (tReadData->getVVCounter() > 0)
    {
        m_iNumTimeSteps = tReadData->m_tViewV[0].header.iNumSteps;
    }
    else if (tReadData->getGeomCounter() > 0)
    {
        m_iNumTimeSteps = tReadData->m_tGeom[0].header.iNumSteps;
    }
    else
    {
        m_iNumTimeSteps = 1;
#ifdef DEBUG
        cerr << "no step" << endl;
#endif
    }
    m_iNumCorners = new int[m_iNumTimeSteps];
    m_iNumPoints = new int[m_iNumTimeSteps];
    m_iNumPolygons = new int[m_iNumTimeSteps];
    for (int i = 0; i < m_iNumTimeSteps; i++)
    {
        m_iNumCorners[i] = 0;
        m_iNumPoints[i] = 0;
        m_iNumPolygons[i] = 0;
    }

    drawPolygons();
    /*   if (!poly->objectOk())
        Covise::sendInfo("polygon is not ok");
        pt_outPort->setCurrentObject(poly);
   */

    drawResults();

    // put results to ports
    if (tReadData->m_bResults && m_iNumTimeSteps == 1)
    {
        for (int i = 0; i < NUMPORTS; i++)
        {
            int iChoice = pt_choice[i]->getValue();
            if (tReadData->m_iPort[iChoice] == 0) // scalar data
            {
                if (S3DRes[i]->objectOk())
                {
                    S3DRes[i]->addAttribute("SPECIES",
                                            tReadData->m_cDescription[iChoice]);
                    pt_outResults[i]->setCurrentObject(S3DRes[i]);
                }
                else
                {
                    Covise::sendError("S3DResult is not ok");
                    return STOP_PIPELINE;
                }
            }
            else // vector data
            {
                if (V3DRes[i]->objectOk())
                {
                    V3DRes[i]->addAttribute("SPECIES", tReadData->m_cDescription[iChoice]);
                    pt_outResults[i]->setCurrentObject(V3DRes[i]);
                }
                else
                {
                    Covise::sendError("V3DResults is not ok");
                    return STOP_PIPELINE;
                }
            }
        }
    }

    tReadData->deleteAll();

    delete[] m_iNumCorners;
    delete[] m_iNumPoints;
    delete[] m_iNumPolygons;
    return SUCCESS;
}

//==============================================================================

FILE *ReadVTF::openFile(const char *filename)
{
    FILE *fp;

// open the obj file
#ifdef WIN32
    if ((fp = Covise::fopen(filename, "rb")) == NULL)
#else
    if ((fp = Covise::fopen(filename, "r")) == NULL)
#endif
    {
        Covise::sendError("ERROR: Can't open file >> %s", filename);
        return (NULL);
    }
    else
    {
#ifdef DEBUGMODE
        fprintf(stderr, "File %s open\n", filename);
#endif
        return (fp);
    }
}

//==============================================================================

int ReadVTF::getInfoAboutFile()
{
    FILE *pt_file = NULL;
    tReadData->m_iNumNodes = 0;
    tReadData->m_iNumElem = 0;
    tReadData->m_iNumRes = 0;
    int iType = 0;
    int iHeaderSize = 0;
    int iDataSize = 0;
    int i;

    Covise::sendInfo("==============================================================================");
    Covise::sendInfo("Open VTF-File: %s", (char *)m_cPath);

    if ((pt_file = openFile((char *)m_cPath)))
    {
        tReadData->reset();
        if (!tReadData->readHeader(pt_file))
        {
            Covise::sendError("Can't read file %s", m_cPath);
            return STOP_PIPELINE;
        }
        else
        {
            // there are 3 bytes between header and the first Block
            for (i = 0; i < 3; i++)
                tReadData->readNextIntElement(pt_file);

            // read the rest of the file
            while (!tReadData->eof()) //!feof(pt_file) )
            {
                iType = tReadData->getBlockType(pt_file);

                switch (iType)
                {

                case 1001:
                    Covise::sendInfo("Found Node Block");
                    tReadData->m_iNumNodes++;
                    break;
                case 1006:
                    Covise::sendInfo("Found IFS Block");
                    tReadData->m_iNumElem++;
                    break;
                case 1007:
                    Covise::sendInfo("Found Element Block");
                    tReadData->m_iNumElem++;
                    break;
                case 1008:
                    Covise::sendInfo("Found GLVIEWGEOMETRY Block");
                    break;
                case 1009:
                    Covise::sendInfo("Found Results Block");
                    tReadData->m_iNumRes++;
                    break;
                case 1010:
                    Covise::sendInfo("Found GLVIEWSCALAR Block");
                    break;
                case 1011:
                    Covise::sendInfo("Found GLVIEWVECTOR Block");
                    break;
                case 1013:
                    Covise::sendInfo("Found TRANSFORMATIONS Block");
                    break;

                case 1026:
                    Covise::sendInfo("Found Transformation Result Block");
                    break;
                case 1027:
                    Covise::sendInfo("Found GLview Transformation Block");
                    break;
                default:
                    if (iType > 1000 && iType < 2000)
                    {
                        Covise::sendInfo("Block with ID %d ignored!!!", iType);
                    }
                    break;
                }

                switch (iType)
                {
                case 1010: //GLVIEWSCALAR:
                    tReadData->readGLViewScalarHeader(pt_file);
                    break;
                case 1011: //GLVIEWVECTOR:
                    tReadData->readGLViewVectorHeader(pt_file);
                    break;

                case 1001:
                case 1006:
                case 1007:
                case 1008:
                case 1009:
                case 1013:
                case 1014:
                case 1016:
                case 1021:
                case 1023:
                case 1024:
                case 1025:
                case 1026:
                case 1027:
                case 1028:
                case 1029:
                case 1030:
                case 1031:
                case 1032:
                case 1033:
                    // BlockID
                    iType = tReadData->getBlockType(pt_file);
                    iHeaderSize = tReadData->getBlockType(pt_file);
                    iDataSize = tReadData->getBlockType(pt_file);
                    // fseek(pt_file, iHeaderSize+iDataSize-4, SEEK_CUR
                    tReadData->seek(iHeaderSize + iDataSize - 2 * sizeof(int));
                    break;
                default:
                    //!!feof(pt_file) ) //!tReadData->eof())
                    while (iType != -999 && iType != -47474747 && !tReadData->eof())
                    {
                        iType = tReadData->getBlockType(pt_file);

                        tReadData->seek(-3);
                        /*int offset=(int)sizeof(int);
                       offset*=-1;
                       offset++;
                       tReadData->seek(offset);*/
                    }
                    break;
                }
            }
            Covise::sendInfo("end of reading header!");
        }
    }
    else
    {
        Covise::sendInfo("Error opening file %s", m_cPath);
        return 1;
    }

    tReadData->reset();
    fclose(pt_file);
    return 0;
}

//==============================================================================

int ReadVTF::readFile()
{
    FILE *pt_file = NULL;
    int iError = 1;

    if ((pt_file = openFile((char *)m_cPath)))
    {
        tReadData->reset();
        if (!tReadData->readHeader(pt_file))
        {
            Covise::sendError("Can't read file %s", m_cPath);
            return 1;
        }
        else
        {
            // there are 3 bytes between header and the first Block
            for (int i = 0; i < 3; i++)
                tReadData->readNextIntElement(pt_file);
            // read the rest of the file

            while (!tReadData->eof()) //!feof(pt_file)
            {
                int iType = tReadData->getBlockType(pt_file);

                switch (iType)
                {
                case 1001: //NODES:
                    iError = tReadData->readNodes(pt_file);
                    if (iError)
                        return 1;
                    break;
                case 1006: //INDEXEDFACESET:
                    //tReadData->readIndexedFaceSet(pt_file);
                    iError = tReadData->readElement(pt_file, true);
                    if (iError)
                        return 1;
                    break;

                case 1007: //ELEMENTS:
                    iError = tReadData->readElement(pt_file);
                    if (iError)
                        return 1;
                    break;
                case 1008: //GLVIEWGEOMETRY:
                    tReadData->readGLViewGeometry(pt_file);
                    break;
                case 1009: //RESULTS:
                    iError = tReadData->readResult(pt_file);
                    if (iError)
                        return 1;
                    break;
                case 1010: //GLVIEWSCALAR:
                    tReadData->readGLViewScalar(pt_file);
                    break;
                case 1011: //GLVIEWVECTOR:
                    tReadData->readGLViewVector(pt_file);
                    break;
                case 1013: //TRANSFORMATIONS:
                    tReadData->readTransformations(pt_file);
                    break;
                case 1014: //VIEWPOINTS:
                    tReadData->readViewPoints(pt_file);
                    break;
                case 1016: //PLOTSERIES:
                    tReadData->readPloteries(pt_file);
                    break;
                case 1021: //USER:
                    tReadData->readUser(pt_file);
                    break;
                case 1023: //POSITIONRESULTS:
                    tReadData->readPositionResults(pt_file);
                    break;
                case 1024: //GLVIEWPOSITIONSCALAR:
                    tReadData->readPositionScalar(pt_file);
                    break;
                case 1025: //GLVIEWPOSITIONVECTOR:
                    tReadData->readGLViewPositionVector(pt_file);
                    break;
                case 1026: //TRANSFORMATIONRESULT:
                    tReadData->readTransformationResult(pt_file);
                    break;
                case 1027: //GLVIEWTRANSFORMATION:
                    tReadData->readGLViewTransformation(pt_file);
                    break;
                case 1028:
                case 1029:
                case 1030:
                case 1031:
                case 1032:
                case 1033:
                {
                    // BlockID
                    iType = tReadData->getBlockType(pt_file);
                    int iHeaderSize = tReadData->getBlockType(pt_file);
                    int iDataSize = tReadData->getBlockType(pt_file);
                    //fseek(pt_file, iHeaderSize+iDataSize-4, SEEK_CUR
                    tReadData->seek(iHeaderSize + iDataSize - 2 * sizeof(int));
                }
                break;
                default:
                    // !feof(pt_file) ) //!tReadData->eof())
                    while (iType != -999 && iType != -47474747 && !tReadData->eof())
                        iType = tReadData->getBlockType(pt_file);
                    break;
                }
            }
            Covise::sendInfo("end of file!");
        }
    }
    else
    {
        Covise::sendInfo("Error opening file %s", m_cPath);
        return 1;
    }
    tReadData->reset();
    fclose(pt_file);
    return 0;
}

//==============================================================================
int ReadVTF::getUSGType(int iType)
{
    int iRet;
    switch (iType)
    {
    case 1:
        iRet = TYPE_POINT;
        break;
    case 2:
        iRet = TYPE_BAR;
        break;
    case 3:
        iRet = TYPE_TRIANGLE;
        break;
    case 4:
        iRet = TYPE_TETRAHEDER; //TYPE_QUAD;
        break;
    case 6:
        iRet = TYPE_PRISM;
        break;
    case 8:
        iRet = TYPE_HEXAEDER;
        break;
    case 15:
        iRet = TYPE_PRISM;
        break;
    case 20:
        iRet = TYPE_HEXAEDER;
        break;
    default:
        Covise::sendError("wrong unstructured grid type: %d", iType);
        iRet = -1;
        break;
    }
    return iRet;
}

//==============================================================================

void ReadVTF::drawPolygons()
{
    bool is_poly = false;
    char objname[100];
    obj = new coDistributedObject *[m_iNumTimeSteps + 1];
    for (int timestep = 0; timestep < m_iNumTimeSteps; timestep++)
    {
        bool firstBlock = true;
        // variables for one element block
        // number of node IDs in a single polygon
        int iNumPoints = 0;
        // number of polygons of the same type in this block
        int iNumPoly = 0;
        // number of element types in the block
        int iNumTypes = 0;
        // parameters for coDoPolygons

        float **xcoord;
        float **ycoord;
        float **zcoord;
        int num_corners = 0;
        int *cornerlist;
        int num_polygons = 0;
        int *polygonlist;
        int *typelist;
        int num_points = 0;

        // cornerToBlock[x] : block to which x belongs
        int *cornerToBlock = new int[tReadData->getNumPoints()];

        // lists
        pt_integerList corner_list = new IntegerList;
        pt_integerList end_corner_list = corner_list;
        pt_integerList polygon_list = new IntegerList;
        pt_integerList end_polygon_list = polygon_list;
        pt_floatList x_coord = new FloatList;
        pt_floatList end_x_coord = x_coord;
        pt_floatList y_coord = new FloatList;
        pt_floatList end_y_coord = y_coord;
        pt_floatList z_coord = new FloatList;
        pt_floatList end_z_coord = z_coord;
        pt_integerList type_list = new IntegerList;
        pt_integerList end_type_list = type_list;

        int iPolyCounter = 0;

        int iElemCounter = tReadData->m_tGeom[0].dataHeader[timestep].iNumElementBlocks;
        int iNodesCounter = tReadData->getNodesCounter();
        int i = 0, j;
        int iActElemBlockID = 0;
        int actElement = 0;
        int iActTransformBlockID = 0;

        for (i = 0; i < iElemCounter; i++)
        {
            iActElemBlockID = tReadData->m_tGeom[0].data[timestep].iElemBlockID[i];

            //search in all elements for the element with the ID iActElementBlockID
            actElement = -1;
            for (int x = 0; x < tReadData->m_iNumElem; x++)
            {
                if (tReadData->m_tElem[x].ID == iActElemBlockID)
                {
                    actElement = x;
                    break;
                }
            }

            if (actElement == -1)
            {
                Covise::sendError("can't find an element with ID %d", iActElemBlockID);
                return;
            }

            //search for a transformation for this element
            iActTransformBlockID = findTransformation(iActElemBlockID, timestep);

            if (sel(i))
            {
                // search the right points for this NodeID
                bool bFlag = false;
                int iNodeBlockID = -1;
                int iNodeNum = tReadData->m_tElem[actElement].header.iNodeBlockID;
                while (!bFlag)
                {
                    iNodeBlockID++;
                    if (tReadData->m_tNodes[iNodeBlockID].ID == iNodeNum)
                        bFlag = true;
                    if (iNodeBlockID > iNodesCounter)
                        Covise::sendInfo("can't find Node ID");
                }
                bFlag = false;

                float M[12]; //transformation matrix

                float x = 0;
                float y = 0;
                float z = 0;

                for (int n = 0; n < tReadData->m_tNodes[iNodeBlockID].header.iNumNodes; n++)
                {
                    if (iActTransformBlockID != -1) //if a transformation matrix was found use it...
                    {
                        memcpy(M, tReadData->m_tTransRes[iActTransformBlockID].pfTransformationMatrix, 12 * sizeof(float));
                        float pointX = tReadData->m_tNodes[iNodeBlockID].point[n].x;
                        float pointY = tReadData->m_tNodes[iNodeBlockID].point[n].y;
                        float pointZ = tReadData->m_tNodes[iNodeBlockID].point[n].z;

                        x = pointX * M[0] + pointY * M[3] + pointZ * M[6] + M[9];
                        y = pointX * M[1] + pointY * M[4] + pointZ * M[7] + M[10];
                        z = pointX * M[2] + pointY * M[5] + pointZ * M[8] + M[11];
                    }
                    else //otherwise use the normal coordinates
                    {
                        x = tReadData->m_tNodes[iNodeBlockID].point[n].x;
                        y = tReadData->m_tNodes[iNodeBlockID].point[n].y;
                        z = tReadData->m_tNodes[iNodeBlockID].point[n].z;
                    }
                    if (num_points == 0)
                    {
                        x_coord->data = x;
                        x_coord->next = NULL;
                        end_x_coord = x_coord;

                        y_coord->data = y;
                        y_coord->next = NULL;
                        end_y_coord = y_coord;

                        z_coord->data = z;
                        z_coord->next = NULL;
                        end_z_coord = z_coord;
                    }
                    else
                    {
                        putFloatListElemAtEnd(end_x_coord, x);
                        end_x_coord = end_x_coord->next;

                        putFloatListElemAtEnd(end_y_coord, y);
                        end_y_coord = end_y_coord->next;

                        putFloatListElemAtEnd(end_z_coord, z);
                        end_z_coord = end_z_coord->next;
                    }

                    if (tReadData->m_tNodes[iNodeBlockID].header.iWithID)
                    {
                        int iNode_ID = tReadData->m_tNodes[iNodeBlockID].point[n].ID;
                        tReadData->setPosInPointList(iNodeBlockID, iNode_ID, num_points);
                    }
                    else
                    {
                        if (tReadData->m_tNodes[iNodeBlockID].point[n].iPosInPointList == -1)
                            tReadData->m_tNodes[iNodeBlockID].point[n].iPosInPointList = num_points;
                    }
                    num_points++;
                }

                iNumTypes = tReadData->m_tElem[actElement].header.iNumElementTypes;

                for (int j = 0; j < iNumTypes; j++)
                {
                    int type = tReadData->m_tElem[actElement].data[j].dataHeader.eType;
                    is_poly = (type == ReadVTFData::IFS);
                    iNumPoly = tReadData->m_tElem[actElement].data[j].dataHeader.iNumElements;

                    num_polygons += iNumPoly;
                    int iNumPointsLast = 0;

                    for (int k = 0; k < iNumPoly; k++)
                    {
                        iNumPointsLast = iNumPoints;
                        iNumPoints = tReadData->m_tElem[actElement].data[j].tElement[k].iNumNodes;

                        if (firstBlock)
                        {
                            polygon_list->data = 0;
                            polygon_list->next = NULL;
                            end_polygon_list = polygon_list;
                            type_list->data = (is_poly) ? 0 : getUSGType(iNumPoints);
                            type_list->next = NULL;
                            end_type_list = type_list;
                            firstBlock = false;
                        }
                        else
                        {
                            int iValue = 0;
                            if (1) //k==0)
                                iValue = end_polygon_list->data + iNumPointsLast;
                            else
                                iValue = end_polygon_list->data + iNumPoints;
                            putIntListElemAtEnd(end_polygon_list, iValue);
                            end_polygon_list = end_polygon_list->next;

                            int iTypeValue = (is_poly) ? 0 : getUSGType(iNumPoints);
                            putIntListElemAtEnd(end_type_list, iTypeValue);
                            end_type_list = end_type_list->next;
                        }

                        if (tReadData->m_tElem[actElement].data[j].tElement[k].iPosInPolyList == -1)
                        {
                            tReadData->m_tElem[actElement].data[j].tElement[k].iPosInPolyList = iPolyCounter;

                            iPolyCounter++;
                        }
                        // we have the node, now we have to copy the points

                        for (int l = 0; l < iNumPoints; l++)
                        {
                            int iPointID = tReadData->m_tElem[actElement].data[j].tElement[k].iNodeID[l];
                            //int iWithID = tReadData->m_tNodes[iNodeBlockID].header.iWithID;
                            int iValue = tReadData->getPosInPointList(iNodeBlockID, iPointID);
                            if (num_corners == 0)
                            {
                                corner_list->data = iValue;
                                corner_list->next = NULL;
                                end_corner_list = corner_list;
                            }
                            else
                            {
                                putIntListElemAtEnd(end_corner_list, iValue);
                                end_corner_list = end_corner_list->next;
                            }
                            if (tReadData->m_iTransCounter > 0)
                            {
                                cornerToBlock[iValue] = i;
                            }
                            num_corners++;
                        }
                    }
                }
            }
        }

        cornerlist = new int[num_corners];
        polygonlist = new int[num_polygons];
        typelist = new int[num_polygons];
        xcoord = new float *[(m_iNumTimeSteps > 0) ? m_iNumTimeSteps : 1];
        ycoord = new float *[(m_iNumTimeSteps > 0) ? m_iNumTimeSteps : 1];
        zcoord = new float *[(m_iNumTimeSteps > 0) ? m_iNumTimeSteps : 1];
        for (i = 0; i < m_iNumTimeSteps || i == 0; i++)
        {
            xcoord[i] = new float[num_points];
            ycoord[i] = new float[num_points];
            zcoord[i] = new float[num_points];
        }

        pt_integerList pt_corner = corner_list;
        pt_integerList pt_poly = polygon_list;
        pt_integerList pt_type = type_list;
        pt_integerList pt_helpInt;
        pt_floatList pt_x = x_coord;
        pt_floatList pt_y = y_coord;
        pt_floatList pt_z = z_coord;
        pt_floatList pt_helpFloat;

        for (i = 0; i < num_corners; i++)
        {
            cornerlist[i] = pt_corner->data;
            pt_corner = pt_corner->next;
        }

        for (i = 0; i < num_polygons; i++)
        {
            polygonlist[i] = pt_poly->data;
            pt_poly = pt_poly->next;
            typelist[i] = pt_type->data;
            pt_type = pt_type->next;
        }

        for (i = 0; i < num_points; i++)
        {
            xcoord[0][i] = pt_x->data;
            pt_x = pt_x->next;

            ycoord[0][i] = pt_y->data;
            pt_y = pt_y->next;

            zcoord[0][i] = pt_z->data;
            pt_z = pt_z->next;
        }

        // Transform
        if (tReadData->m_iTransCounter > 0)
        {
            for (j = 0; j < num_points; j++)
            {
                transformCoords(&xcoord[0][j], &ycoord[0][j], &zcoord[0][j], cornerToBlock[j], timestep, is_poly);
            }
        }

        // delete all
        for (i = 0; i < num_corners; i++)
        {
            if (corner_list == NULL)
            {
                Covise::sendError("Error deleting corner_list");
            }
            pt_helpInt = corner_list;
            corner_list = corner_list->next;
            delete pt_helpInt;
        }

        for (i = 0; i < num_polygons; i++)
        {
            if (polygon_list == NULL)
            {
                Covise::sendError("Error deleting polygon_list");
            }
            pt_helpInt = polygon_list;
            polygon_list = pt_helpInt->next;
            delete pt_helpInt;
        }

        for (i = 0; i < num_polygons; i++)
        {
            if (type_list == NULL)
            {
                Covise::sendError("Error deleting type_list");
            }
            pt_helpInt = type_list;
            type_list = pt_helpInt->next;
            delete pt_helpInt;
        }

        for (i = 0; i < num_points; i++)
        {
            if (x_coord == NULL)
            {
                Covise::sendError("Error deleting xCoord list");
            }
            pt_helpFloat = x_coord;
            x_coord = pt_helpFloat->next;
            delete pt_helpFloat;
        }

        for (i = 0; i < num_points; i++)
        {
            if (y_coord == NULL)
            {
                Covise::sendError("Error deleting xCoord list");
            }
            pt_helpFloat = y_coord;
            y_coord = pt_helpFloat->next;
            delete pt_helpFloat;
        }

        for (i = 0; i < num_points; i++)
        {
            if (z_coord == NULL)
            {
                Covise::sendError("Error deleting zCoord list");
            }
            pt_helpFloat = z_coord;
            z_coord = pt_helpFloat->next;
            delete pt_helpFloat;
        }

        coDistributedObject *out;
        if (m_iNumTimeSteps > 1)
        {

            sprintf(objname, "%s_timestep_%d", (is_poly) ? pt_outPoly->getObjName() : pt_outPort->getObjName(), timestep);
            if (!is_poly)
            {
                poly = new coDoUnstructuredGrid(objname, num_polygons, num_corners, num_points, polygonlist,
                                                cornerlist, xcoord[0], ycoord[0], zcoord[0], typelist);
                out = poly;
            }
            else
            {
                out = new coDoPolygons(objname, num_points, xcoord[0], ycoord[0], zcoord[0],
                                       num_corners, cornerlist, num_polygons, polygonlist);
                out->addAttribute("vertexOrder", "2");
            }
            obj[timestep] = out;

            obj[timestep + 1] = NULL;
        }
        else
        {
            if (!is_poly)
            {
                poly = new coDoUnstructuredGrid(pt_outPort->getObjName(), num_polygons,
                                                num_corners, num_points, polygonlist,
                                                cornerlist, xcoord[0], ycoord[0], zcoord[0], typelist);
                pt_outPort->setCurrentObject(poly);
            }
            else
            {
                out = new coDoPolygons(pt_outPoly->getObjName(), num_points, xcoord[0], ycoord[0], zcoord[0],
                                       num_corners, cornerlist, num_polygons, polygonlist);
                out->addAttribute("vertexOrder", "2");
                pt_outPoly->setCurrentObject(out);
            }
        }
        m_iNumCorners[timestep] = num_corners;
        m_iNumPoints[timestep] = num_points;
        m_iNumPolygons[timestep] = iPolyCounter;

        delete[] cornerlist;
        delete[] polygonlist;
        for (i = 0; i < m_iNumTimeSteps || i == 0; i++)
        {
            delete[] xcoord[i];
            delete[] ycoord[i];
            delete[] zcoord[i];
        }
        delete[] xcoord;
        delete[] ycoord;
        delete[] zcoord;
        delete[] cornerToBlock;
    }
    if (m_iNumTimeSteps > 1)
    {
        sprintf(objname, "1 %d", m_iNumTimeSteps);
        coDoSet *set = new coDoSet((is_poly) ? pt_outPoly->getObjName() : pt_outPort->getObjName(), obj);
        set->addAttribute("TIMESTEP", objname);
        (is_poly) ? pt_outPoly->setCurrentObject(set) : pt_outPort->setCurrentObject(set);
    }
}

//==============================================================================

pt_integerList ReadVTF::newIntListElement(int data)
{
    pt_integerList pt_list;
    pt_list = new IntegerList;
    if (pt_list != NULL)
    {
        pt_list->data = data;
        pt_list->next = NULL;
    }
    else
        Covise::sendInfo("problems allocate space for LISTPTR");
    return pt_list;
}

//==============================================================================

bool ReadVTF::putIntListElemAtEnd(pt_integerList last, int data)
{
    pt_integerList pt_newElem;
    bool bRet = false;
    pt_newElem = newIntListElement(data);
    if (pt_newElem == NULL)
        bRet = false;
    else
    {
        last->next = pt_newElem;
        bRet = true;
    }
    return bRet;
}

//==============================================================================

pt_floatList ReadVTF::newFloatListElement(float data)
{
    pt_floatList pt_list;
    pt_list = new FloatList;
    if (pt_list != NULL)
    {
        pt_list->data = data;
        pt_list->next = NULL;
    }
    else
        Covise::sendInfo("problems allocate space for LISTPTR");
    return pt_list;
}

//==============================================================================

bool ReadVTF::putFloatListElemAtEnd(pt_floatList last, float data)
{
    pt_floatList pt_newElem;
    bool bRet = false;
    pt_newElem = newFloatListElement(data);
    if (pt_newElem == NULL)
        bRet = false;
    else if (last != NULL)
    {
        last->next = pt_newElem;
        bRet = true;
    }
    else
        Covise::sendInfo("can't put new int element at the end of list");
    return bRet;
}

//==============================================================================

void ReadVTF::deleteIntListElem(pt_integerList ptr)
{
    pt_integerList pt_help = ptr;
    ptr = pt_help->next;
    delete ptr;
}

//==============================================================================

void ReadVTF::deleteFloatListElem(pt_floatList ptr)
{
    pt_floatList pt_help = ptr;
    ptr = pt_help->next;
    delete ptr;
}

//==============================================================================

void ReadVTF::drawResults()
{
    int iElemNumScalar = 0;
    int iElemNumVector = 0;
    int iDimension = 0;
    int iNumResults = 0;
    int iResCounter = tReadData->getResCounter();
    int iMappingType = 0;
    int iPos = 0;
    int iWithID;
    int i = 0;
    int j = 0;
    int r, s, t = 0, u, blockid;

    // flags
    bool bScalarPerElementFace = false;
    bool bScalarPerElem = false;
    bool bVectorPerElem = false;
    bool bScalarPerNode = false;
    bool bVectorPerNode = false;

    // ==========results per node========
    // lists for vector and scalar data
    float **ElementFace_scalar_data = new float *[m_iNumTimeSteps];
    float **Node_scalar_data = new float *[m_iNumTimeSteps];
    float **Node_xVector_data = new float *[m_iNumTimeSteps];
    float **Node_yVector_data = new float *[m_iNumTimeSteps];
    float **Node_zVector_data = new float *[m_iNumTimeSteps];

    if (Node_scalar_data == NULL || Node_xVector_data == NULL || Node_yVector_data == NULL || Node_zVector_data == NULL)
    {
        return;
    }

    // allocate space
    for (i = 0; i < m_iNumTimeSteps; i++)
    {
        ElementFace_scalar_data[i] = NULL;
        Node_scalar_data[i] = NULL;
        Node_xVector_data[i] = new float[m_iNumPoints[i]];
        Node_yVector_data[i] = new float[m_iNumPoints[i]];
        Node_zVector_data[i] = new float[m_iNumPoints[i]];
        // initialize arrays
        for (j = 0; j < m_iNumPoints[i]; j++)
        {
            Node_xVector_data[i][j] = 0.0;
            Node_yVector_data[i][j] = 0.0;
            Node_zVector_data[i][j] = 0.0;
        }
    }

    // ==========results per element========
    // lists for vector and scalar data

    char objname[100];
    int k = 0;
    int n = 0;

    float **Elem_scalar_data;
    Elem_scalar_data = new float *[m_iNumTimeSteps];

    float ***Elem_Coord_data; // x, y, z data
    Elem_Coord_data = new float **[m_iNumTimeSteps];

    for (i = 0; i < m_iNumTimeSteps; i++)
        Elem_Coord_data[i] = new float *[3];

    for (i = 0; i < m_iNumTimeSteps; i++)
        for (j = 0; j < 3; j++)
            Elem_Coord_data[i][j] = new float[m_iNumPolygons[i]];

    // allocate space
    for (i = 0; i < m_iNumTimeSteps; i++)
        Elem_scalar_data[i] = new float[m_iNumPolygons[i]];

    // initialize arrays
    for (i = 0; i < m_iNumTimeSteps; i++)
        for (j = 0; j < m_iNumPolygons[i]; j++)
            Elem_scalar_data[i][j] = 0.0;

    for (n = 0; n < m_iNumTimeSteps; n++)
        for (i = 0; i < 3; i++)
            for (j = 0; j < m_iNumPolygons[n]; j++)
                Elem_Coord_data[n][i][j] = 0.0;

    int **Scalar_Node_or_Elem;
    int **Vector_Node_or_Elem;
    Scalar_Node_or_Elem = new int *[m_iNumTimeSteps];
    Vector_Node_or_Elem = new int *[m_iNumTimeSteps];
    /* // only for testing
   // to check results, put them into a file
   ofstream one ("/vobs/covise/src/application/general/READ_VTF/timestepOne.data");
   ofstream two ("/vobs/covise/src/application/general/READ_VTF/timestepTwo.data");
   */
    // let's begin working...:-)
    for (r = 0; r < 2; r++) //
    {
        for (s = 0; (r == 0) ? s < tReadData->m_iVSCounter : s < tReadData->m_iVVCounter; s++) // every scalar field
        {
            for (t = 0; (r == 0) ? t < tReadData->m_tViewS[s].header.iNumSteps : t < tReadData->m_tViewV[s].header.iNumSteps; t++)
            {
                // lists where I want to remember, on which position there are results per node or per element
                // I'll write in this list: Per node: 0, per element: 1

                Scalar_Node_or_Elem[t] = new int[m_iNumPoints[t]];
                Vector_Node_or_Elem[t] = new int[m_iNumPoints[t]];
                // initialize these arrays
                for (i = 0; i < m_iNumPoints[t]; i++)
                {
                    Scalar_Node_or_Elem[t][i] = 0;
                    Vector_Node_or_Elem[t][i] = 0;
                }
                for (u = 0; (r == 0) ? u < tReadData->m_tViewS[s].dataHeader[t].iNumResultsBlocks :
                                     // assume blocks in right order
                                u < tReadData->m_tViewV[s].dataHeader[t].iNumResultsBlocks;
                     u++)
                {
                    if (sel(u))
                    {
                        blockid = (r == 0) ? tReadData->m_tViewS[s].data[t].iResultBlockID[u] : tReadData->m_tViewV[s].data[t].iResultBlockID[u];
                        for (i = 0; i < iResCounter; i++)
                        {
                            if (tReadData->m_tRes[i].ID == blockid)
                            {
                                break;
                            }
                        }

                        if (i >= iResCounter)
                        {
                            cerr << "Error" << endl;
                            i = iResCounter - 1;
                        }

                        iNumResults = tReadData->m_tRes[i].header.iNumResults;
                        iDimension = tReadData->m_tRes[i].header.iDimension;
                        iMappingType = tReadData->m_tRes[i].header.iMappingType;
                        int nodeNumber = 0;
                        for (int nodeN = 0; nodeN < tReadData->getNodesCounter(); nodeN++)
                        {
                            if (tReadData->m_tNodes[nodeN].ID == tReadData->m_tRes[i].header.iMapToBlockID)
                            {
                                nodeNumber = nodeN;
                                break;
                            }
                        }
                        iWithID = tReadData->m_tRes[i].header.iWithID;

                        //============ results per node ================
                        if (iDimension == 1) // scalar data
                        {
                            if (iMappingType == 0) // results per node
                            {
                                Node_scalar_data[t] = new float[m_iNumPoints[t]];
                                bScalarPerNode = true;
                                for (int j = 0; j < iNumResults; j++)
                                {
                                    if (iWithID)
                                    {
                                        int iPointID = tReadData->m_tRes[i].data[j].iUserID;
                                        iPos = tReadData->getPosInPointList(nodeNumber, iPointID);
                                    }
                                    else
                                    {
                                        iPos = tReadData->m_tNodes[nodeNumber].point[j].iPosInPointList;
                                    }
                                    Node_scalar_data[t][iPos] = tReadData->m_tRes[i].data[j].fScalarValue;
                                    Scalar_Node_or_Elem[t][iPos] = 0;
                                    //iNodeNumScalar++;
                                    /* // only for testing
                              if (m_iNumTimeSteps==1)
                              if (tReadData->m_tRes[i].data[j].fScalarValue)
                              one<<tReadData->m_tRes[i].data[j].fScalarValue<<endl;
                           */
                                }
                            }
                        }
                        else // vector data
                        {
                            if (iMappingType == 0) // results per node
                            {
                                bVectorPerNode = true;
                                for (int j = 0; j < iNumResults; j++)
                                {
                                    if (iWithID)
                                    {
                                        int iPointID = tReadData->m_tRes[i].data[j].iUserID;
                                        iPos = tReadData->getPosInPointList(nodeNumber, iPointID);
                                    }
                                    else
                                        iPos = tReadData->m_tNodes[nodeNumber].point[j].iPosInPointList;

                                    Node_xVector_data[t][iPos] = tReadData->m_tRes[i].data[j].fVectorValue[0];
                                    Node_yVector_data[t][iPos] = tReadData->m_tRes[i].data[j].fVectorValue[1];
                                    Node_zVector_data[t][iPos] = tReadData->m_tRes[i].data[j].fVectorValue[2];
                                    Vector_Node_or_Elem[t][iPos] = 0;
                                    //iNodeNumVector++;
                                    /* // only for testing
                              if (m_iNumTimeSteps==1)
                              if (tReadData->m_tRes[i].data[j].fVectorValue[0])
                              {
                              one<<tReadData->m_tRes[i].data[j].fVectorValue[0]
                              <<" "<<tReadData->m_tRes[i].data[j].fVectorValue[1]<<
                              " "<<tReadData->m_tRes[i].data[j].fVectorValue[2]<<endl;

                              }
                           */
                                }
                            }
                        }
                        //============= results per element ====================
                        if (iDimension == 1) // scalar data
                        {
                            if (iMappingType == 1) // results per element
                            {
                                bScalarPerElem = true;
                                int iNumElemTypes = tReadData->m_tElem[nodeNumber].header.iNumElementTypes;
                                // int iCounter = 0;
                                for (int k = 0; k < iNumElemTypes; k++)
                                {
                                    //test
                                    int iTest = tReadData->m_tElem[nodeNumber].data[k].dataHeader.iNumElements;
                                    int min = (iTest > iNumResults) ? iNumResults : iTest;
                                    for (int n = 0; n < m_iNumTimeSteps; n++)
                                    {
                                        //iNumResults
                                        for (int j = 0; j < min; j++)
                                        {
                                            if (iWithID)
                                            {
                                                int iUserID = tReadData->m_tRes[i].data[j].iUserID;
                                                //cerr << "witID" << endl;
                                                iPos = tReadData->getPosInPolyList(nodeNumber, iUserID);
                                            }
                                            else
                                            {
                                                iPos = tReadData->m_tElem[nodeNumber].data[k].tElement[j].iPosInPolyList;
                                            }
                                            if (iPos < m_iNumPolygons[n])
                                                Elem_scalar_data[n][iPos] = tReadData->m_tRes[i].data[j].fScalarValue;
                                            if (iPos < m_iNumPoints[t])
                                            {
                                                Scalar_Node_or_Elem[t][iPos] = 1;
                                                iElemNumScalar++;
                                            }
                                            /* // only for testing
                                    if (n==0)
                                    {
                                    if (tReadData->m_tRes[i].data[j].fScalarValue)
                                    one<<tReadData->m_tRes[i].data[j].fScalarValue<<endl;
                                    }
                                    else
                                    {
                                    if (tReadData->m_tRes[i].data[j].fScalarValue)
                                    two<<tReadData->m_tRes[i].data[j].fScalarValue<<endl;
                                    }
                                 */
                                        }
                                    }
                                }
                            }
                        }
                        else // vector data
                        {
                            if (iMappingType == 1) // results per element
                            {
                                bVectorPerElem = true;
                                int iNumElemTypes = tReadData->m_tElem[nodeNumber].header.iNumElementTypes;
                                for (int k = 0; k < iNumElemTypes; k++)
                                {
                                    for (int n = 0; n < m_iNumTimeSteps; n++)
                                    {
                                        for (int j = 0; j < iNumResults; j++)
                                        {
                                            if (iWithID)
                                            {
                                                int iUserID = tReadData->m_tRes[i].data[j].iUserID;
                                                iPos = tReadData->getPosInPolyList(nodeNumber, iUserID);
                                            }
                                            else
                                            {
                                                iPos = tReadData->m_tElem[nodeNumber].data[k].tElement[j].iPosInPolyList;
                                                iElemNumVector++;
                                            }
                                            for (int m = 0; m < 3; m++)
                                                Elem_Coord_data[n][m][iPos] = tReadData->m_tRes[i].data[j].fVectorValue[m];
                                            Vector_Node_or_Elem[t][iPos] = 1;
                                            iElemNumVector++;
                                            /* // only for testing
                                    if (n==0)
                                    {
                                    if (tReadData->m_tRes[i].data[j].fVectorValue[0])
                                    {
                                    one<<tReadData->m_tRes[i].data[j].fVectorValue[0]
                                    <<" "<<tReadData->m_tRes[i].data[j].fVectorValue[1]
                                    <<" "<<tReadData->m_tRes[i].data[j].fVectorValue[2]<<endl;
                                    }
                                    }
                                    else
                                    {
                                    if (tReadData->m_tRes[i].data[j].fVectorValue[0])
                                    {
                                    two<<tReadData->m_tRes[i].data[j].fVectorValue[0]
                                    <<" "<<tReadData->m_tRes[i].data[j].fVectorValue[1]
                                    <<" "<<tReadData->m_tRes[i].data[j].fVectorValue[2]<<endl;
                                    }
                                    }
                                 */
                                        }
                                    }
                                }
                            }
                        }
                        if (iDimension == 1 || iDimension == 3)
                        {
                            if (iMappingType == 4) // results per element face, not yet implemented
                            {
                                Covise::sendInfo("results per element face not yet implemented");
                                ElementFace_scalar_data[t] = new float[iNumResults];
                                bScalarPerElementFace = true;
                            }

                            if (iMappingType == 2)
                            {
                                Covise::sendInfo("results per face!!");
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // now we want to put the data to the ports...
    // but perhaps we have results per node AND per element...
    // than we must put these results together!

    // list for both scalar data
    float **Both_scalar_data[NUMPORTS];
    for (i = 0; i < NUMPORTS; i++)
        Both_scalar_data[i] = new float *[m_iNumTimeSteps];
    for (i = 0; i < NUMPORTS; i++)
        for (j = 0; j < m_iNumTimeSteps; j++)
            Both_scalar_data[i][j] = new float[m_iNumPoints[j]];

    // initialize this array
    for (i = 0; i < NUMPORTS; i++)
        for (j = 0; j < m_iNumTimeSteps; j++)
            for (k = 0; k < m_iNumPoints[j]; k++)
                Both_scalar_data[i][j][k] = 0.0;

    float **Scalar_elem_to_node[NUMPORTS]; // new lists with results per node
    int *iNumPointsScalar[NUMPORTS];
    int *iNumPointsScalarElem[NUMPORTS];
    coCellToVert cellToVert;

    for (i = 0; i < NUMPORTS; i++)
    {
        Scalar_elem_to_node[i] = NULL;
        iNumPointsScalar[i] = new int[m_iNumTimeSteps];

        for (j = 0; j < m_iNumTimeSteps; j++)
            iNumPointsScalar[i][j] = iNumResults;

        iNumPointsScalarElem[i] = new int[m_iNumTimeSteps];
    }

    // first we are looking after the scalar data:
    if (bScalarPerNode && bScalarPerElem)
    {
        // map results per element to results per node
        float **scalar_data; // new lists with results per node
        scalar_data = new float *[m_iNumTimeSteps];
        int iCounterOne = 0;
        int iCounterTwo = 0;
        for (i = 0; i < NUMPORTS; i++)
        {
            for (int n = 0; n < m_iNumTimeSteps; n++)
            {
                //cout<<"scalar data per node and element:"<<endl;
                //cout<<"timestep "<<n<<endl;
                sprintf(objname, "%s_A_%d", pt_outResults[i]->getObjName(), n);
                coDoFloat *scal = (coDoFloat *)cellToVert.interpolate(obj[n], 1,
                                                                      iNumPointsScalarElem[i][n],
                                                                      Elem_scalar_data[n], NULL,
                                                                      NULL, objname);

                scal->getAddress(&scalar_data[n]);
                // put lists together
                // first copy list with results per node in new list
                for (int j = 0; j < m_iNumPoints[n]; j++)
                {
                    if (Scalar_Node_or_Elem[t][j] == 0)
                    {
                        Both_scalar_data[i][n][j] = Node_scalar_data[n][iCounterOne];
                        iCounterOne++;
                    }
                    else
                    {
                        Both_scalar_data[i][n][j] = scalar_data[n][iCounterTwo];
                        iCounterTwo++;
                    }
                }
            }
        }
        /*if (iCounterOne != m_iNumPoints[i])
        Covise::sendInfo("Not all Node_scalar_data have been read!");*/

        delete[] scalar_data;
    }
    else if (bScalarPerElem)
    {
        // map results per elemet to results per node
        for (i = 0; i < NUMPORTS; i++)
        {
            Scalar_elem_to_node[i] = new float *[m_iNumTimeSteps];
            for (int n = 0; n < m_iNumTimeSteps; n++)
            {
                //cout<<"scalar data per element:"<<endl;
                //cout<<"timestep "<<n<<endl;
                sprintf(objname, "%s_B_%d", pt_outResults[i]->getObjName(), n);
                coDoFloat *scal = (coDoFloat *)cellToVert.interpolate(obj[n], 1,
                                                                      iNumPointsScalar[i][n],
                                                                      Elem_scalar_data[n], NULL,
                                                                      NULL, objname);

                if (scal)
                {
                    iNumPointsScalar[i][n] = scal->getNumPoints();
                    scal->getAddress(&Scalar_elem_to_node[i][n]);
                }
                else
                {
                    Covise::sendError("File contains both blocks with and without data. Please read either blocks with or without data.");
                    Scalar_elem_to_node[i][n] = NULL;
                }
            }
        }
    }
    else if (bScalarPerElementFace)
    {
        // map results per face to results per node, not yet implemented
    }

    // lists for both vector data
    float **Both_xVector_data[NUMPORTS];
    float **Both_yVector_data[NUMPORTS];
    float **Both_zVector_data[NUMPORTS];

    for (i = 0; i < NUMPORTS; i++)
    {
        Both_xVector_data[i] = new float *[m_iNumTimeSteps];
        Both_yVector_data[i] = new float *[m_iNumTimeSteps];
        Both_zVector_data[i] = new float *[m_iNumTimeSteps];
    }

    for (i = 0; i < NUMPORTS; i++)
        for (j = 0; j < m_iNumTimeSteps; j++)
        {
            Both_xVector_data[i][j] = new float[m_iNumPoints[j]];
            Both_yVector_data[i][j] = new float[m_iNumPoints[j]];
            Both_zVector_data[i][j] = new float[m_iNumPoints[j]];
        }

    // initialize these arrays
    for (i = 0; i < NUMPORTS; i++)
        for (int j = 0; j < m_iNumTimeSteps; j++)
            for (k = 0; k < m_iNumPoints[j]; k++)
            {
                Both_xVector_data[i][j][k] = 0.0;
                Both_yVector_data[i][j][k] = 0.0;
                Both_zVector_data[i][j][k] = 0.0;
            }

    float **vector_xElem_to_node[NUMPORTS]; // new list with results per node
    float **vector_yElem_to_node[NUMPORTS];
    float **vector_zElem_to_node[NUMPORTS];

    for (i = 0; i < NUMPORTS; i++)
    {
        vector_xElem_to_node[i] = new float *[m_iNumTimeSteps];
        vector_yElem_to_node[i] = new float *[m_iNumTimeSteps];
        vector_zElem_to_node[i] = new float *[m_iNumTimeSteps];
    }

    int iNumPointsVec[NUMPORTS];
    int iNumPointsVecElem[NUMPORTS];
    // now we are looking after the vector data
    if (bVectorPerNode && bVectorPerElem)
    {
        // map results per elemet to results per node
        float *vector_xData[NUMPORTS]; // new list with results per node
        float *vector_yData[NUMPORTS];
        float *vector_zData[NUMPORTS];

        /*for (i=0; i<NUMPORTS; i++)
        {
        vector_xData[i] = new float* [m_iNumTimeSteps];
        vector_yData[i] = new float* [m_iNumTimeSteps];
        vector_zData[i] = new float* [m_iNumTimeSteps];
        }*/
        int iCounterOne = 0;
        int iCounterTwo = 0;
        for (i = 0; i < NUMPORTS; i++)
        {
            for (int n = 0; n < m_iNumTimeSteps; n++)
            {
                //cout<<"vetro data per node and element:"<<endl;
                //cout<<"timestep "<<n<<endl;
                sprintf(objname, "%s_C_%d", pt_outResults[i]->getObjName(), n);
                V3DResNew[i] = (coDoVec3 *)cellToVert.interpolate(obj[n], 3,
                                                                  iNumPointsVecElem[i],
                                                                  Elem_Coord_data[n][0],
                                                                  Elem_Coord_data[n][1],
                                                                  Elem_Coord_data[n][2],
                                                                  objname);
                V3DResNew[i]->getAddresses(&vector_xData[i], &vector_yData[i],
                                           &vector_zData[i]);
                // put lists together
                // first copy list with results per node in new list
                for (int j = 0; j < m_iNumPoints[n]; j++)
                {
                    if (Scalar_Node_or_Elem[n][j] == 0)
                    {
                        Both_xVector_data[i][n][j] = Node_xVector_data[n][iCounterOne];
                        Both_yVector_data[i][n][j] = Node_yVector_data[n][iCounterOne];
                        Both_zVector_data[i][n][j] = Node_zVector_data[n][iCounterOne];
                        iCounterOne++;
                    }
                    else
                    {
                        Both_xVector_data[i][n][j] = *vector_xData[iCounterTwo];
                        Both_yVector_data[i][n][j] = *vector_yData[iCounterTwo];
                        Both_zVector_data[i][n][j] = *vector_zData[iCounterTwo];
                        iCounterTwo++;
                    }
                }
            }
        }
        /*if (iCounterOne != m_iNumPoints)
        Covise::sendInfo("Not all Node_vector_data have been read!");*/
    }
    else if (bVectorPerElem)
    {
        char objname[100];
        for (i = 0; i < NUMPORTS; i++)
        {
            for (int n = 0; n < m_iNumTimeSteps; n++)
            {
                //cout<<"vector data per element:"<<endl;
                //cout<<"timestep "<<n<<endl;
                sprintf(objname, "%s_D_%d", pt_outResults[i]->getObjName(), n);
                V3DResNew[i] = (coDoVec3 *)cellToVert.interpolate(obj[n], 3,
                                                                  iNumPointsVec[i],
                                                                  Elem_Coord_data[n][0],
                                                                  Elem_Coord_data[n][1],
                                                                  Elem_Coord_data[n][2], objname);
                iNumPointsVec[i] = V3DResNew[i]->getNumPoints();
                V3DResNew[i]->getAddresses(&vector_xElem_to_node[i][n],
                                           &vector_yElem_to_node[i][n],
                                           &vector_zElem_to_node[i][n]);
            }
        }
    }
    // put data to ports
    coDistributedObject **objects = new coDistributedObject *[m_iNumTimeSteps + 1];

    for (i = 0; i < NUMPORTS; i++)
    {
        int iChoice = pt_choice[i]->getValue();

        for (n = 0; n < m_iNumTimeSteps; n++)
        {
            objects[n] = NULL;

            if (m_iNumTimeSteps > 1)
                sprintf(objname, "%s_E_%d", pt_outResults[i]->getObjName(), n);
            else
                sprintf(objname, "%s", pt_outResults[i]->getObjName());
            S3DRes[i] = NULL;
            V3DRes[i] = NULL;
            if (tReadData->m_iPort[iChoice] == 0) // scalar data
            {
                if (bScalarPerNode && bScalarPerElem)
                {
#ifdef DEBUG
                    Covise::sendInfo("Scalar data per node and scalar data per element are used");
#endif
                    S3DRes[i] = new coDoFloat(objname, m_iNumPoints[n],
                                              Both_scalar_data[i][n]);
                }
                else if (bScalarPerNode)
                {
#ifdef DEBUG
                    Covise::sendInfo("ScalarPerNode");
#endif
                    S3DRes[i] = new coDoFloat(objname, m_iNumPoints[n],
                                              Node_scalar_data[n]);
                }
                else if (bScalarPerElem)
                {
#ifdef DEBUG
                    Covise::sendInfo("ScalarPerElem");
#endif
                    if (Scalar_elem_to_node[i][n])
                        S3DRes[i] = new coDoFloat(objname, iNumPointsScalar[i][n],
                                                  Scalar_elem_to_node[i][n]);
                }
                else if (bScalarPerElementFace) // not yet implemented
                {
#ifdef DEBUG
                    Covise::sendInfo("ScalarPerElementFace");
#endif
                }
                if (m_iNumTimeSteps > 1)
                    objects[n] = S3DRes[i];
            }
            else // vector data
            {
                if (bVectorPerNode && bVectorPerElem)
                {
#ifdef DEBUG
                    Covise::sendInfo("Vector data per node and vector data per element are used");
#endif
                    V3DRes[i] = new coDoVec3(objname, m_iNumPoints[n],
                                             Both_xVector_data[i][n],
                                             Both_yVector_data[i][n],
                                             Both_zVector_data[i][n]);
                }
                if (bVectorPerNode)
                {
#ifdef DEBUG
                    Covise::sendInfo("VectorPerNode");
#endif
                    V3DRes[i] = new coDoVec3(objname, m_iNumPoints[n],
                                             Node_xVector_data[n],
                                             Node_yVector_data[n],
                                             Node_zVector_data[n]);
                }
                else if (bVectorPerElem)
                {
#ifdef DEBUG
                    Covise::sendInfo("VectorPerElem");
#endif
                    V3DRes[i] = new coDoVec3(objname, iNumPointsVec[i],
                                             vector_xElem_to_node[i][n],
                                             vector_yElem_to_node[i][n],
                                             vector_zElem_to_node[i][n]);
                }
                if (m_iNumTimeSteps > 1)
                    objects[n] = V3DRes[i];
            }
        }
        if (m_iNumTimeSteps > 1 && objects[0] != NULL)
        {
            objects[n] = NULL;
            coDoSet *set = new coDoSet(pt_outResults[i]->getObjName(), objects);
            pt_outResults[i]->setCurrentObject(set);
        }
    }
    // delete all

    for (i = 0; i < m_iNumTimeSteps; i++)
    {
        delete[] Node_scalar_data[i];
        delete[] Node_xVector_data[i];
        delete[] Node_yVector_data[i];
        delete[] Node_zVector_data[i];
        delete[] Elem_scalar_data[i];
        delete[] ElementFace_scalar_data[i];
        delete[] Scalar_Node_or_Elem[i];
        delete[] Vector_Node_or_Elem[i];
    }

    delete[] Node_scalar_data;
    delete[] Node_xVector_data;
    delete[] Node_yVector_data;
    delete[] Node_zVector_data;
    delete[] ElementFace_scalar_data;

    for (i = 0; i < m_iNumTimeSteps; i++)
        for (j = 0; j < 3; j++)
            delete[] Elem_Coord_data[i][j];

    delete[] Scalar_Node_or_Elem;
    delete[] Vector_Node_or_Elem;

    for (i = 0; i < NUMPORTS; i++)
        for (j = 0; j < m_iNumTimeSteps; j++)
        {
            delete[] Both_scalar_data[i][j];
            delete[] Both_xVector_data[i][j];
            delete[] Both_yVector_data[i][j];
            delete[] Both_zVector_data[i][j];
        }

    for (i = 0; i < NUMPORTS; i++)
    {
        delete[] Scalar_elem_to_node[i];
        delete[] iNumPointsScalar[i];
        delete[] iNumPointsScalarElem[i];
    }

    for (i = 0; i < NUMPORTS; i++)
    {
        delete[] vector_xElem_to_node[i];
        delete[] vector_yElem_to_node[i];
        delete[] vector_zElem_to_node[i];
    }
}

/*
 * returns the ID of the TransformationResultBlock for the given element ID, at
 * the given time step
 */
int ReadVTF::findTransformation(int elementID, int timeStep)
{
    (void)timeStep;
    coModule::sendInfo("Find transformation for element ID %d", elementID);
    int transBlockID = -1;

    for (int j = 0; j < tReadData->m_iTransResCounter; j++)
    {
        if (tReadData->m_tTransRes[j].header.iElementBlockID == elementID)
        {
            transBlockID = j;
            coModule::sendInfo("Transformation found");
            break;
        }
    }

    return transBlockID;
}

void ReadVTF::transformCoords(float *x, float *y, float *z, int block, int step, bool IFS)
{
    float xs = *x;
    float ys = *y;
    float zs = *z;
    int transBl = 0;

    //                           [ux vx wx]
    //  [ x y z ] = [x y z 1]  * [uy vy wy]
    //                           [uz vz wz]
    //                           [ox oy oz]

    float ox = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][9] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][9];
    float oy = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][10] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][10];
    float oz = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][11] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][11];

    float ux = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][0] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][0];
    float uy = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][3] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][3];
    float uz = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][6] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][6];

    float vx = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][1] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][1];
    float vy = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][4] : tReadData->m_tTrans[transBl].data[step].iElementTransMatrix[block][4];
    float vz = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][7] : tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][7];

    float wx = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][2] : tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][2];
    float wy = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][5] : tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][5];
    float wz = (IFS) ? tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][8] : tReadData->m_tTrans[transBl].data[step].iIFSTransMatrix[block][8];

    *x = ox + xs * ux + ys * uy + zs * uz;
    *y = oy + xs * vx + ys * vy + zs * vz;
    *z = oz + xs * wx + ys * wy + zs * wz;
}

MODULE_MAIN(IO, ReadVTF)
