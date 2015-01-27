/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*********************************************************************************\
 **                                                       (C)2008 VHLRS         **
 **                                                                             **
 ** Description: ReadBifBof Plugin                                              **
 **              Reads in bif and bof files                                     **
 **              Output: points or unstructured grid  and polygons for 2d data  **
 **                                                                             **
 ** Author: M. Theilacker                                                       **
 **                                                                             **
 ** History:                                                                    **
 **                                                                             **
 **                                                                             **
 **                                                                             **
\*********************************************************************************/

#include "ReadBIFBOF.h"
#include <config/CoviseConfig.h>
/*! \brief constructor
 *
 * create In/Output Ports and module parameters here
 */
ReadBIFBOF::ReadBIFBOF(int argc, char **argv)
    : coModule(argc, argv, "Read BIF and BOF data")
{
    p_bifFileParam = addFileBrowserParam("BIFFile", "BIF-File");
    p_bofFileParam = addFileBrowserParam("BOFFile", "BOF-File");
    p_3D_gridOutPort = addOutputPort("GridOut_3D", "UnstructuredGrid", "3D Grid-Unstructured");
    p_2D_gridOutPort = addOutputPort("GridOut_2D", "Polygons", "2D Grid-Polygons");
    p_VectorData = addOutputPort("vectorData", "Vec3", "Vector Data");
    p_ScalarData = addOutputPort("skalarData", "Float", "Scalar Data");

    p_bofFileParam->setValue(".", "*");

    string CatalogFile = coCoviseConfig::getEntry(
        "value", "Module.BifBof.CatalogFile", "share/covise/bifbof_dscat.ds");
    char absolutePath[1000];
    getname(absolutePath, CatalogFile.c_str());
    bifBof = new BifBof(absolutePath);
    if (bifBof->getLastError() != 0)
    {
        Covise::sendError("Error initializing DSIO Library, probably wrong path to dscat file current file:");
        Covise::sendError("%s", absolutePath);
    }
    readingComplete = 0;

    // IDs for geoElements
    BifGeoElements::makeGeoIDs();
}

ReadBIFBOF::~ReadBIFBOF()
{
    delete bifBof;
}
void ReadBIFBOF::param(const char * /* name */, bool /* inMapLoading */)
{
}

int ReadBIFBOF::compute(const char * /* port */)
{

    coDoUnstructuredGrid *grid_out;
    coDoPolygons *poly_out;
    coDoPoints *points_out;
    //DO_Unstructured_S3D_Data *coviseFloatObject;
    coDoFloat *coviseFloatObject;
    coDoVec3 *coviseVecObject;
    BifNodalPoints *coordinates = NULL;
    vector<BifGeoElements *> geoElements;
    BofScalarData *bofScalarData;
    BofVectorData *bofVectorData;
    int num_types = 0, num_elem = 0, num_poly = 0, num_corner = 0, num_conn = 0, num_coord = 0, num_scalar_coord = 0, num_vector_coord = 0;
    int *elem_list = NULL, *conn_list = NULL, *poly_list = NULL, *type_list = NULL, *corner_list = NULL;
    vector<int> *elem_vec, *conn_vec, *type_vec, *corner_vec, *poly_vec;
    float *x_coords = NULL, *y_coords = NULL, *z_coords = NULL;
    const char *bif_filename = NULL;
    const char *bof_filename = NULL;

    bif_filename = p_bifFileParam->getValue();
    bof_filename = p_bofFileParam->getValue();

    BifGeoElements::clear(); //reset all vectors

    if (!bif_filename)
    {
        Covise::sendError("File not readable!");
        return STOP_PIPELINE;
    }

    int err = bifBof->openBifFile(bif_filename);
    if (err != 0)
        cout << "!!!!!!!! Error from DSIO:    " << bifBof->returnErrorCode(err) << endl;
    string programName, date, time, description;
    bifBof->readFileHeader(programName, date, time, description);

    Covise::sendInfo("ProgramName: %s Date: %s Time: %s Description: %s", programName.c_str(), date.c_str(), time.c_str(), description.c_str());

    for (int seqElemNum = 1; bifBof->getLastError() == 0 || bifBof->getLastError() == BIFBOF_UNKNOWN_DATATYPE; seqElemNum++)
    {
        int id, subheaderFlag, numRecords;
        if (bifBof->readElementHeader(headBuffer, seqElemNum, id, subheaderFlag, numRecords) == 0)
        {
            // number of Coordinates in record
            int numCoords = getHeaderIntWord(4);

            // Data element with record headers whose
            // record header must be read first
            //          if ( subheaderFlag == 1 )
            //          {
            //             seqElemNum--;
            //          }
            // RESTART, CGM and DEDEF are not read
            /*else */ if (subheaderFlag < -23)
            {
                //back to start
            }
            else if (subheaderFlag >= 0)
            {
                ///subheaderFlag < 0 -> irregular data element

                //
                // Read nodal points
                //
                if (id == BifElement::NODALPOINTS)
                {
                    int minID = getHeaderIntWord(21); //min Node ID
                    int maxID = getHeaderIntWord(22); //max Node ID
                    coordinates = new BifNodalPoints(id, minID, maxID, numCoords, bifBof);
                    int ret = coordinates->readInCoordinates(readingComplete);
                    if (ret != 0)
                        return handleError(); //"DSRREC"
                    num_coord = coordinates->getNumCoordinates();
                    // get the coordinates list
                    ret = coordinates->getCoviseCoordinates(&x_coords, &y_coords, &z_coords);
                    if (ret != 0)
                        return handleError();
                }

                //
                // Read element data
                //
                if ((BifGeoElements::geoIDs).find(id) != (BifGeoElements::geoIDs).end())
                {
                    // number of elements in record
                    int numRec = getHeaderIntWord(4);

                    geoElements.push_back(new BifGeoElements(id, numRec, bifBof));

                    int ret = geoElements.back()->readInConnections(coordinates, readingComplete);

                    if (ret != 0)
                        return handleError(); //"DSRREC"
                }

                // readingComplete == 0   -> Next record
                // readingComplete == 1   -> Next header
                // else                   -> Next data element
                if (readingComplete == 1)
                {
                    // Next header
                    seqElemNum = seqElemNum - 1;
                }
            }
        }
    }

    bifBof->closeBifFile();
    // done reading file

    // get geoElement data structures
    conn_vec = BifGeoElements::getCoviseConnections();
    elem_vec = BifGeoElements::getCoviseElementList();
    type_vec = BifGeoElements::getCoviseTypeList();
    corner_vec = BifGeoElements::getCoviseCornerList();
    poly_vec = BifGeoElements::getCovisePolyList();

    if (x_coords == NULL || y_coords == NULL || z_coords == NULL)
    {
        Covise::sendError("No coordinates in file");
        return STOP_PIPELINE;
    }

    // making data structures for COVISE
    num_elem = BifGeoElements::getNumElements();
    num_poly = BifGeoElements::getNumPolys();
    num_types = BifGeoElements::getNumTypes();
    elem_list = new int[num_elem];
    poly_list = new int[num_poly];
    type_list = new int[num_types];

    for (int i = 0; i < num_elem; i++)
    {
        elem_list[i] = elem_vec->at(i);
        type_list[i] = type_vec->at(i);
    }
    for (int i = 0; i < num_poly; i++)
    {
        poly_list[i] = poly_vec->at(i);
    }
    num_conn = BifGeoElements::getNumConnections();
    num_corner = BifGeoElements::getNumCorners();
    conn_list = new int[num_conn];
    corner_list = new int[num_corner];

    for (int i = 0; i < num_conn; i++)
    {
        conn_list[i] = conn_vec->at(i);
    }
    for (int i = 0; i < num_corner; i++)
    {
        corner_list[i] = corner_vec->at(i);
    }
    // send points if no unstructured or polygon data
    // else send triangle polygons
    if (elem_list == NULL || conn_list == NULL || type_list == NULL)
    {
        points_out = new coDoPoints(p_3D_gridOutPort->getObjName(), num_coord,
                                    x_coords, y_coords, z_coords);
        p_3D_gridOutPort->setCurrentObject(points_out);
        cout << "points in file:" << num_coord << endl;
    }
    else
    {
        grid_out = new coDoUnstructuredGrid(p_3D_gridOutPort->getObjName(),
                                            num_elem, num_conn, num_coord,
                                            elem_list, conn_list,
                                            x_coords, y_coords, z_coords, type_list);
        //3d data
        grid_out->addAttribute("SIMULATION", bif_filename);
        p_3D_gridOutPort->setCurrentObject(grid_out);
        poly_out = new coDoPolygons(p_2D_gridOutPort->getObjName(),
                                    num_coord, x_coords, y_coords, z_coords,
                                    num_corner, corner_list, num_poly, poly_list);
        poly_out->addAttribute("SIMULATION", bif_filename);
        //2d data
        p_2D_gridOutPort->setCurrentObject(poly_out);
        cout << "number of 3d elements:" << num_elem << " number of 2d elements:" << num_poly << endl;
    }

    cout << "BIF***********************************************************" << endl;
    //-----------------------------------------------
    if (bof_filename)
    {
        bifBof->openBifFile(bof_filename);

        std::string programName, date, time, description;
        bifBof->readFileHeader(programName, date, time, description);
        Covise::sendInfo("ProgramName: %s Date: %s Time: %s Description: %s", programName.c_str(), date.c_str(), time.c_str(), description.c_str());

        for (int seqElemNum = 1; bifBof->getLastError() == 0 || bifBof->getLastError() == BIFBOF_UNKNOWN_DATATYPE; seqElemNum++)
        {
            int id, subheaderFlag, numRecords;
            if (bifBof->readElementHeader(headBuffer, seqElemNum, id, subheaderFlag, numRecords) == 0)
            {
                if (subheaderFlag >= 0)
                {
                    ///subheaderFlag < 0 -> irregular data element

                    //
                    // Read temperatures
                    //

                    if (id == BifElement::TEMP)
                    {
                        // start and end ids for nodes
                        int minID = getHeaderIntWord(21);
                        int maxID = getHeaderIntWord(22);
                        int tempUnit = getHeaderIntWord(23);

                        bofScalarData = new BofScalarData(id, coordinates->getMinCoordID(), coordinates->getMaxCoordID(), coordinates->getNumCoordinates(), minID, maxID, bifBof);

                        int ret = bofScalarData->readScalarData(coordinates, readingComplete);

                        if (ret != 0)
                            return handleError(); //"DSRREC"

                        num_scalar_coord = bofScalarData->getNumCoordinates();
                        // get the coordinates list
                        scalarValues = bofScalarData->getScalarArray();

                        // reading the temperatur unit defined in headerword 23
                        std::string tmpStr;
                        switch (tempUnit)
                        {
                        case 1:
                            tmpStr = "[C]";
                            break;
                        case 2:
                            tmpStr = "[K]";
                            break;
                        case 3:
                            tmpStr = "[F]";
                            break;
                        case 4:
                            tmpStr = "[Reaumur]";
                            break;
                        }
                        //
                        speciesName = "Temperatur";
                        speciesName += tmpStr;
                    }
                    //---------------------
                    if (id == BifElement::DEFO)
                    {
                        // start and end ids for nodes
                        int minID = getHeaderIntWord(21);
                        int maxID = getHeaderIntWord(22);
                        int DatypFlg = getHeaderIntWord(23);

                        bofVectorData = new BofVectorData(id, coordinates->getMinCoordID(), coordinates->getMaxCoordID(), coordinates->getNumCoordinates(), minID, maxID, bifBof);

                        int ret = bofVectorData->readVectorData(coordinates, readingComplete);

                        if (ret != 0)
                            return handleError(); //"DSRREC"

                        num_vector_coord = bofVectorData->getNumCoordinates();
                        // get the coordinates list
                        xValues = bofVectorData->getXArray();
                        yValues = bofVectorData->getYArray();
                        zValues = bofVectorData->getZArray();
                        // reading the Data Type defined in headerword 23
                        switch (DatypFlg)
                        {
                        case 0:
                            speciesName = "displacement";
                            break;
                        case 1:
                            speciesName = "velocity";
                            break;
                        case 2:
                            speciesName = "acceleration";
                            break;
                        }
                    }
                    //-----------------------

                    // readingComplete == 0   -> Next record
                    // readingComplete == 1   -> Next header
                    // else                   -> Next data element
                    if (readingComplete == 1)
                    {
                        // Next header
                        seqElemNum = seqElemNum - 1;
                        //soll: schleife1
                    }
                }
            }
        }
        // done reading file
        bifBof->closeBifFile();
    }
    else
    {
        //         send_error("File not readable!");
        return STOP_PIPELINE;
    }

    // send points if no triangle data
    // else send polygons
    if (num_vector_coord != 0)
    {
        coviseVecObject = new coDoVec3(p_VectorData->getObjName(),
                                       num_vector_coord, xValues, yValues, zValues);
        coviseVecObject->addAttribute("SPECIES", speciesName.c_str());
        p_VectorData->setCurrentObject(coviseVecObject);
    }

    if (scalarValues != NULL)
    {
        coviseFloatObject = new coDoFloat(p_ScalarData->getObjName(),
                                          num_scalar_coord, scalarValues);
        coviseFloatObject->addAttribute("SPECIES", speciesName.c_str());
        p_ScalarData->setCurrentObject(coviseFloatObject);
    }
    cout << "BOF***********************************************************" << endl;
    //________________________________________________

    return 0;
}

int ReadBIFBOF::handleError()
{
    return bifBof->getLastError();
}

//
// get the int value of header word num (of dsele20)
//
int ReadBIFBOF::getHeaderIntWord(int num)
{
    return headBuffer[num - 1].i;
}

//
// get the float value of header word num (of dsele20)
//
float ReadBIFBOF::getHeaderFloatWord(int num)
{
    return headBuffer[num - 1].f;
}

//
// get the char value of header word num (of dsele20)
//
char *ReadBIFBOF::getHeaderCharWord(int num)
{
    return headBuffer[num - 1].c;
}

/*MODULE_MAIN ( IO, ReadBIFBOF )
int main(int argc, char *argv[])
{
   ReadBIFBOF *application = new ReadBIFBOF;
   application->start(argc,argv);
}*/

MODULE_MAIN(IO, ReadBIFBOF)
