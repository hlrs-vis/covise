/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2005 GER  **
 **                                                                        **
 ** Description: Read ASCII Files.					   **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     	 Oliver Krause			           **
 **     			Zentrum fuer angewandte Informatik Koeln   **
 **                     University  of Cologne                             **
 **                                                   			   **
 **                                                                        **
 ** Creation Date: May 2005                                                **
\**************************************************************************/

#include <api/coModule.h>
#include <limits.h>
#include <float.h>
#include "ReadAscii.h"
#include <string>
#include <fstream>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>

#define BUFINC 1

// little helper for debugging
void hello(const string str = " ");
void hello(const int);

// Constructor
ReadASCII::ReadASCII(int argc, char **argv)
    : coSimpleModule(argc, argv, "Read files to create points (w or w/o scalar or vector values) or a grid.")
{
    // Create ports:
    poGeometry = addOutputPort("Geometry", "Points|UniformGrid|RectilinearGrid|StructuredGrid", "Points or Grid");
    poGeometry->setInfo("Points or Grid");

    poScalar = addOutputPort("ScalarData", "Float", "Scalar data");
    poScalar->setInfo("Scalar data");

    poVector = addOutputPort("VectorData", "Vec3", "Vector data");
    poVector->setInfo("Vector data");

    // Create parameters:
    pbrFile = addFileBrowserParam("FilePath", "File of interest");
    pbrFile->setValue("data/", "*");
    pBufferSize = addInt32Param("MaxLineLength", "Length of the max line in the file");
    pBufferSize->setValue(80);

    pGeom = addChoiceParam("GeometryType", "Geometry type of the input data");
    char *pGeomLabel[4];
    pGeomLabel[0] = (char *)"Points";
    pGeomLabel[1] = (char *)"Uniform Grid";
    pGeomLabel[2] = (char *)"Rectiliniar Grid";
    pGeomLabel[3] = (char *)"Structured Grid";
    pGeom->setValue(4, pGeomLabel, 0);

    pHeader = addBooleanParam("Header", "Does the file contain an header?");
    pHeader->setValue(false);

    /* pSedPattern = addStringParam("SedPattern", "Pattern to scan and manipulate the input file");
      pSedPattern->setValue("insert pattern here");
   */
    pHeaderByteOffset = addInt32Param("HeaderByteOffset", "Offset for the header in byte");
    pHeaderByteOffset->setValue(0);

    pHeaderLineSkip = addInt32Param("HeaderLineSkip", "Offset for the header in lines");
    pHeaderLineSkip->setValue(0);

    pDimPattern = addStringParam("DimPattern", "Pattern to scan the dimensions");
    pDimPattern->setValue("dimensions = %d %d %d");

    pDimX = addInt32Param("DimX", "X Dimension");
    pDimX->setValue(0);

    pUniDistX = addFloatParam("UniDistX", "Uniform distance in X direction");
    pUniDistX->setValue(-1);

    pDimY = addInt32Param("DimY", "Y Dimension");
    pDimY->setValue(0);

    pUniDistY = addFloatParam("UniDistY", "Uniform distance in Y direction");
    pUniDistY->setValue(-1);

    pDimZ = addInt32Param("DimZ", "Z Dimension");
    pDimZ->setValue(0);

    pUniDistZ = addFloatParam("UniDistZ", "Uniform distance in Z direction");
    pUniDistZ->setValue(-1);

    pInterl = addBooleanParam("DataInterleaving", "Is the data input in an interleaved format?");
    pInterl->setValue(false);

    pPointsNum = addInt32Param("NumberOfPoints", "Number of Points");
    pPointsNum->setValue(0);

    pDataByteOffset = addInt32Param("DataByteOffset", "Offset for the data in byte");
    pDataByteOffset->setValue(0);

    pDataLineOffset = addInt32Param("DataLineSkip", "Offset for the data in lines");
    pDataLineOffset->setValue(0);

    pDataFormat = addStringParam("DataFormat", "Format to scan the data");
    pDataFormat->setValue("%XP, %YP, %ZP");

    pCoordSequence = addChoiceParam("GridIndexIterationOrder", "Iteration Order of the index (Structured Grid only)");
    char *pCoordSeqLabel[3];
    pCoordSeqLabel[0] = (char *)"i, j, k";
    pCoordSeqLabel[1] = (char *)"k, j, i";
    pCoordSeqLabel[2] = (char *)" ";
    pCoordSequence->setValue(3, pCoordSeqLabel, 1);

    pOutputResult = addBooleanParam("PrintResultsToConsole", "Print the result in Format: x,y,z(coords) - scalar - x,y,z(value)");
    pOutputResult->setValue(false);

    pOutputDebug = addBooleanParam("PrintDebugInfoToConsole", "Print used parameters and their values");
    pOutputDebug->setValue(false);

    pScale = addFloatParam("Scale", "ScaleFactor");
    pScale->setValue(1);
}

int ReadASCII::compute(const char *)
{
    // some deklarations
    int num_of_points(-1),
        dimX(pDimX->getValue()),
        dimY(pDimY->getValue()),
        dimZ(pDimZ->getValue());

    // module title is the filename without path
    const char *path = pbrFile->getValue();
    const char *filename = path + strlen(path);
    while (filename >= path)
    {
        if (((*filename == '\\') || (*filename == '/')) && (filename[1] != '\0'))
        {
            filename++;
            break;
        }
        filename--;
    }
    char title[256];
    strcpy(title, "ASCII:");
    strcat(title, filename);
    char *ptr = strstr(title, ".");
    if (ptr && ptr != filename)
        *ptr = '\0';
#ifndef YAC
    setTitle(title);
#endif

    // find out how many points are needed
    printf("pGeom: %d\n", pGeom->getValue());
    switch (pGeom->getValue())
    {
    case 0:
        num_of_points = pPointsNum->getValue();
        break;

    case 1:
        num_of_points = 2;
        break;

    case 2:
        num_of_points = -1;
        break;

    case 3:
        num_of_points = dimX * dimY * dimZ;
        break;
    }
    pPointsNum->setValue(num_of_points);

    // check header byte/line skip
    if ((pHeaderByteOffset->getValue() > 0) && (pHeaderLineSkip->getValue() > 0))
    {
        sendError("Header line AND byte offset cannot be applied at same time.");
        return STOP_PIPELINE;
    }

    // check data byte/line skip
    if ((pDataLineOffset->getValue() > 0) && (pDataByteOffset->getValue() > 0))
    {
        sendError("Data line AND byte offset cannot be applied at same time.");
        return STOP_PIPELINE;
    }

    // expand the pattern
    param(NULL, false);
    string pattern(pDataFormat->getValue());
    expandPattern(pattern, pBufferSize->getValue());

    // pointer for coords and data
    float *x_coords = NULL;
    float *y_coords = NULL;
    float *z_coords = NULL;
    float *x_data = NULL;
    float *y_data = NULL;
    float *z_data = NULL;
    float *scalar_data = NULL;

    // initialize arrays
    if ((pGeom->getValue() != 0) && ((dimX < 0) && (dimY < 0) && (dimZ < 0)))
    {
        sendError("Invalid dimensions. Cannot parse file.");
        return STOP_PIPELINE;
    }

    if (pGeom->getValue() != 2)
    {
        printf("points: %d\n", num_of_points);
        if (num_of_points > 0)
        {
            x_coords = new float[num_of_points];
            y_coords = new float[num_of_points];
            z_coords = new float[num_of_points];
            if (strContains(pattern, "%XV"))
                x_data = new float[num_of_points];
            if (strContains(pattern, "%YV"))
                y_data = new float[num_of_points];
            if (strContains(pattern, "%ZV"))
                z_data = new float[num_of_points];
            if (strContains(pattern, "%S"))
                scalar_data = new float[num_of_points];
        }
        else
        {
            sendError("Too few points. Check parameter settings");
            return STOP_PIPELINE;
        }
    }
    else
    {
        x_coords = new float[dimX];
        y_coords = new float[dimY];
        z_coords = new float[dimZ];
    }

    // fill arrays with zeros
    if (pGeom->getValue() != 2)
    {
        for (int i = 0; i < num_of_points; i++)
        {
            if (x_coords != NULL)
                x_coords[i] = 0;
            if (y_coords != NULL)
                y_coords[i] = 0;
            if (z_coords != NULL)
                z_coords[i] = 0;
            if (x_data != NULL)
                x_data[i] = 0;
            if (y_data != NULL)
                y_data[i] = 0;
            if (z_data != NULL)
                z_data[i] = 0;
            if (scalar_data != NULL)
                scalar_data[i] = 0;
        }
    }
    else
    {
        for (int i = 0; i < 2; i++)
            x_coords[i] = 0;
        for (int i = 0; i < 2; i++)
            y_coords[i] = 0;
        for (int i = 0; i < 2; i++)
            z_coords[i] = 0;
    }

    // if geometry is structured or rectilinear grid, create an uniform grid first
    // the data from file will replace the pre-initialized data
    if ((pGeom->getValue() == 3))
    {
        // create an uniform grid
        if ((dimX > 0) && (dimY > 0) && (dimZ > 0))
        {
            for (int i = 0; i < num_of_points; i++)
            {
                if (pUniDistX->getValue() > 0)
                    x_coords[i] = ((i / (dimY * dimZ) % dimX)) * pUniDistX->getValue();

                if (pUniDistY->getValue() > 0)
                    y_coords[i] = ((i / dimZ) % dimY) * pUniDistY->getValue();

                if (pUniDistZ->getValue() > 0)
                    z_coords[i] = (i % dimZ) * pUniDistZ->getValue();
            }
        }
        else
        {
            sendError("Invalid dimensions.");
            return STOP_PIPELINE;
        }
    }

    // collect all parameter for readout
    params parameter;
    parameter.xcoords = x_coords;
    parameter.ycoords = y_coords;
    parameter.zcoords = z_coords;
    parameter.xdata = x_data;
    parameter.ydata = y_data;
    parameter.zdata = z_data;
    parameter.scalardata = scalar_data;

    // read data from file if the pattern contains an '%'
    if (strContains(pattern, "%"))
    {
        if (!readDataFromFile(parameter))
        {
            sendError("An reading error occured. Data could not been read.");
            delete[] x_coords;
            delete[] y_coords;
            delete[] z_coords;
            delete[] x_data;
            delete[] y_data;
            delete[] z_data;
            delete[] scalar_data;
            return STOP_PIPELINE;
        }
    }

    // scale data
    if (pScale->getValue() != 1)
    {
        if (pGeom->getValue() != 2)
        {
            for (int i = 0; i < num_of_points; i++)
            {
                if (x_coords != NULL)
                    x_coords[i] *= pScale->getValue();
                if (y_coords != NULL)
                    y_coords[i] *= pScale->getValue();
                if (z_coords != NULL)
                    z_coords[i] *= pScale->getValue();
                if (x_data != NULL)
                    x_data[i] *= pScale->getValue();
                if (y_data != NULL)
                    y_data[i] *= pScale->getValue();
                if (z_data != NULL)
                    z_data[i] *= pScale->getValue();
                if (scalar_data != NULL)
                    scalar_data[i] *= pScale->getValue();
            }
        }
        else
        {
            for (int i = 0; i < pDimX->getValue(); i++)
                x_coords[i] *= pScale->getValue();

            for (int i = 0; i < pDimY->getValue(); i++)
                y_coords[i] *= pScale->getValue();

            for (int i = 0; i < pDimZ->getValue(); i++)
                z_coords[i] *= pScale->getValue();
        }
    }

    // create data objects (Points, Grid, ...)
    switch (pGeom->getValue())
    {
    // points, vector data, scalar data
    case 0:
        if ((x_coords != NULL) && (y_coords != NULL) && (z_coords != NULL))
        {
            coDoPoints *points = new coDoPoints(poGeometry->getObjName(), num_of_points, x_coords, y_coords, z_coords);
            poGeometry->setCurrentObject((coDistributedObject *)points);
        }
        if (scalar_data != NULL)
        {
            coDoFloat *US_scalars = new coDoFloat(poScalar->getObjName(), num_of_points, scalar_data);
            poScalar->setCurrentObject((coDistributedObject *)US_scalars);
        }
        if ((x_data != NULL) && (y_data != NULL) && (z_data != NULL))
        {
            coDoVec3 *US_vector = new coDoVec3(poVector->getObjName(), num_of_points, x_data, y_data, z_data);
            poVector->setCurrentObject((coDistributedObject *)US_vector);
        }
        break;
    // uniform grid

    case 1:
        if ((pDimX->getValue() > 0) && (pDimY->getValue() > 0) && (pDimZ->getValue() > 0))
        {
            coDoUniformGrid *uniGrid = new coDoUniformGrid(poGeometry->getObjName(), (int)pDimX->getValue(), (int)pDimY->getValue(), (int)pDimZ->getValue(), x_coords[0], x_coords[1], y_coords[0], y_coords[1], z_coords[0], z_coords[1]);
            poGeometry->setCurrentObject((coDistributedObject *)uniGrid);
        }
        break;

    // rectilinear grid
    case 2:
        if ((pDimX->getValue() > 0) && (pDimY->getValue() > 0) && (pDimZ->getValue() > 0))
        {
            coDoRectilinearGrid *rectGrid = new coDoRectilinearGrid(poGeometry->getObjName(), (int)pDimX->getValue(), (int)pDimY->getValue(), (int)pDimZ->getValue(), x_coords, y_coords, z_coords);
            poGeometry->setCurrentObject((coDistributedObject *)rectGrid);
        }
        break;

    // structured grid
    case 3:
        if ((dimX > 0) && (dimY > 0) && (dimZ > 0))
        {
            coDoStructuredGrid *structGrid = new coDoStructuredGrid(poGeometry->getObjName(), dimX, dimY, dimZ, x_coords, y_coords, z_coords);
            poGeometry->setCurrentObject((coDistributedObject *)structGrid);
        }
        break;

    // something has gone wrong
    default:
        sendError("An error occured. How could this happen?");
        delete[] x_coords;
        delete[] y_coords;
        delete[] z_coords;
        delete[] x_data;
        delete[] y_data;
        delete[] z_data;
        delete[] scalar_data;
        return STOP_PIPELINE;
        break;
    }

    // output debug information to console
    if (pOutputDebug->getValue())
        printDebug();

    // output results to console
    if (pOutputResult->getValue())
        printResult(parameter);

    // clean up
    delete[] x_coords;
    delete[] y_coords;
    delete[] z_coords;
    delete[] x_data;
    delete[] y_data;
    delete[] z_data;
    delete[] scalar_data;

    return CONTINUE_PIPELINE;
}

bool ReadASCII::readDataFromFile(params para)
{
    // deklarations
    int num_of_points(pPointsNum->getValue()),
        dimX(pDimX->getValue()),
        dimY(pDimY->getValue()),
        dimZ(pDimZ->getValue());
    string pattern(pDataFormat->getValue());
    expandPattern(pattern, pBufferSize->getValue());
    char *buffer;

    // is buffer size greater than zero?
    if (pBufferSize->getValue() > 0)
        buffer = new char[pBufferSize->getValue()];
    else
    {
        sendError("Buffer size has to be greater than 0.");
        return false;
    }

    // opening file
    const char *path = pbrFile->getValue();
    ifstream infile(path, ios::in);

    // skip offset
    int offsetCounter(0);
    if (pDataLineOffset->getValue() > 0)
        while ((infile.getline(buffer, pBufferSize->getValue())) && (offsetCounter < pDataLineOffset->getValue() - 1))
            offsetCounter++;
    if (pDataByteOffset->getValue() > 0)
        infile.seekg(pDataByteOffset->getValue(), ios::beg);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // reading the data
    //
    // This is the major part of the module. The data will be parsed with help from the data pattern.
    // - According to the actual geometry is a different storage and different settings nessesary. So the module
    //		works with a <switch> to seperate the cases.
    // - Then we have to seperate 2 further cases: interleaved and non-interleaved data. This is done by an <if> inside
    // 	the actual case
    // - Then we can parse the data.
    // - When parsing the data for structured grids (case 4), the module can handle 2 different sequences of data storage:
    //		x-y-z (non vtk conform) and z-y-x (vtk conform) --> see documentation for further information
    //

    sendInfo("reading data");
    bool error = false;

    switch (pGeom->getValue())
    {
    // points
    case 0:
    {
        //////////////////////////////////////////////////////////////////////////////////
        // read the vector data and store them
        //

        if (num_of_points <= 0)
        {
            sendError("Too few points");
            return false;
        }

        ////////////////////////////////////////////////
        // read a line
        //
        int arrayCounter = 0;
        int resCounter = 0;
        string strTemp;

        // data interleaving or not
        if (pInterl->getValue())
        {
            while ((infile.getline(buffer, pBufferSize->getValue())))
            {
                //helper
                size_t pattern_length(pattern.length());
                unsigned int reading_pos1(0), reading_pos2(0);
                int num(0), res(0), result(0);
                // points
                float x(0), y(0), z(0);
                // data
                float x2(0), y2(0), z2(0);
                float scalar(0);
                // dummy
                float dummy(0);

                ////////////////////////////////////////////////////////
                // The data will be parsed with help from the pattern
                //

                while (reading_pos1 < pattern_length)
                {

                    // handle escape sequenzes
                    if (pattern.at(reading_pos1) == '\\')
                    {
                        switch (pattern.at(reading_pos1 + 1))
                        {
                        case 'n':
                            infile.getline(buffer, pBufferSize->getValue());
                            reading_pos1 += 2;
                            reading_pos2 = 0;
                            break;
                        default:

                            break;
                        }
                    }

                    // parsing the data
                    if (pattern.at(reading_pos1) == '%')
                    {
                        // parse first char
                        switch (pattern.at(reading_pos1 + 1))
                        {
                        case '%':
                            reading_pos1 += 2;
                            reading_pos2 += 2;
                            break;

                        case 'X':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found XP: " << x << ", num: " << num << endl;
                            }
                            else if (pattern.at(reading_pos1 + 2) == 'V')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &x2, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found XV: " << x2 << ", num: " << num << endl;
                            }
                            break;

                        case 'Y':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found YP: " << y << ", num: " << num << endl;
                            }
                            else if (pattern.at(reading_pos1 + 2) == 'V')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &y2, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found YV: " << y2 << ", num: " << num << endl;
                            }
                            break;

                        case 'Z':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found ZP: " << z << ", num: " << num << endl;
                            }
                            else if (pattern.at(reading_pos1 + 2) == 'V')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &z2, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found ZV: " << z2 << ", num: " << num << endl;
                            }
                            break;

                        case 'S':
                            result += sscanf(buffer + reading_pos2, "%f%n", &scalar, &num);

                            if (result > 0)
                                res += result;
                            else
                                error = true;

                            reading_pos2 += num;
                            reading_pos1 += 2;
                            //cout << "found S: " << scalar << ", num: " << num << endl;
                            break;

                        case 'F':
                            result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                            if (result > 0)
                                res += result;
                            else
                                error = true;

                            reading_pos2 += num;
                            reading_pos1 += 2;
                            //cout << "found F: " << dummy << ", num: " << num << endl;
                            break;

                        default:
                            reading_pos1++;
                            reading_pos2++;
                            break;
                        }
                    }
                    else
                    {
                        reading_pos1++;
                        reading_pos2++;
                    }

                } // while(reading_pos1 < pattern_length)

                if ((res > 0) && (arrayCounter < num_of_points))
                {
                    if (para.xcoords != NULL)
                        para.xcoords[arrayCounter] = x;
                    if (para.ycoords != NULL)
                        para.ycoords[arrayCounter] = y;
                    if (para.zcoords != NULL)
                        para.zcoords[arrayCounter] = z;
                    if (para.xdata != NULL)
                        para.xdata[arrayCounter] = x2;
                    if (para.ydata != NULL)
                        para.ydata[arrayCounter] = y2;
                    if (para.zdata != NULL)
                        para.zdata[arrayCounter] = z2;
                    if (para.scalardata != NULL)
                        para.scalardata[arrayCounter] = scalar;

                    resCounter++;
                    arrayCounter++;
                }
            } // while((infile.getline(buffer, pBufferSize->getValue())))
        } // if(!pInterl->getValue())
        else
        {

            while (pattern.length() > 1)
            {

                // The pattern has to be parted into parts seperated from a \n
                size_t pos = pattern.find("\\n", 0);
                if (pos > 0)
                {
                    strTemp = pattern.substr(0, pos);
                    pattern.erase(0, pos + 2);
                }
                else
                {
                    strTemp = pattern;
                    pattern.erase();
                }

                // iteration over the points to read
                for (int i = 0; i < num_of_points; i++)
                {

                    if (infile.getline(buffer, pBufferSize->getValue()))
                    {
                        //cout << i << " - " << strTemp << "==>" << buffer << endl;

                        //helper
                        unsigned int reading_pos1(0), reading_pos2(0);
                        int num(0), res(0), result(0);
                        // points
                        float x(0), y(0), z(0);
                        // data
                        float x2(0), y2(0), z2(0);
                        float scalar(0);
                        // dummy
                        float dummy(0);

                        while (reading_pos1 < strTemp.length())
                        {
                            if (strTemp.at(reading_pos1) == '%')
                            {
                                switch (strTemp.at(reading_pos1 + 1))
                                {
                                case '%':
                                    reading_pos1 += 2;
                                    reading_pos2 += 2;
                                    break;

                                case 'X':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.xcoords[i] = x;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        resCounter++;
                                        //cout << "found XP: " << x << ", num: " << num << endl;
                                    }
                                    else if (strTemp.at(reading_pos1 + 2) == 'V')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &x2, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.xdata[i] = x2;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found XV: " << x2 << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'Y':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.ycoords[i] = y;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found YP: " << y << ", num: " << num << endl;
                                    }
                                    else if (strTemp.at(reading_pos1 + 2) == 'V')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &y2, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.ydata[i] = y2;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found YV: " << y2 << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'Z':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.zcoords[i] = z;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found ZP: " << z << ", num: " << num << endl;
                                    }
                                    else if (strTemp.at(reading_pos1 + 2) == 'V')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &z2, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.zdata[i] = z2;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found ZV: " << z2 << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'S':
                                    result += sscanf(buffer + reading_pos2, "%f%n", &scalar, &num);

                                    if (result > 0)
                                    {
                                        res += result;
                                        para.scalardata[i] = scalar;
                                    }
                                    else
                                        error = true;

                                    reading_pos2 += num;
                                    reading_pos1 += 2;
                                    //cout << "found S: " << scalar << ", num: " << num << endl;
                                    break;

                                case 'F':
                                    result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                                    if (result > 0)
                                        res += result;
                                    else
                                        error = true;

                                    reading_pos2 += num;
                                    reading_pos1 += 2;
                                    //cout << "found F: " << dummy << ", num: " << num << endl;
                                    break;

                                default:
                                    reading_pos1++;
                                    reading_pos2++;
                                    break;

                                } // switch
                            } // if(pattern.at(reading_pos1) == '%')
                            else
                            {
                                reading_pos1++;
                                reading_pos2++;
                            }
                        }
                    } // if(infile.getline(buffer, pBufferSize->getValue()))
                } // for(int i=0; i<num_of_points; i++)
            } // while(pattern.length()>1)

        } // else

        if (error)
            sendError("An error occured while reading. Not all data could be received. Check Data Format and Data Offset.\n");
    }
    break;

    // uniform grid
    case 1:
    {
        param(NULL, false);
        string strTemp;
        int resCounter(0);
        if ((pDimX->getValue()) && (pDimY->getValue()) && (pDimZ->getValue()))
        {
            ///////////////////////////////////////////////////////////////////
            // find {x,y,z}-min and {x,y,z}-max with help from the pattern
            //

            if (pInterl->getValue())
            {
                for (int counter = 0; counter < 2; counter++)
                {
                    infile.getline(buffer, pBufferSize->getValue());

                    //helper
                    size_t pattern_length(pattern.length());
                    unsigned int reading_pos1(0), reading_pos2(0);
                    int num(0), res(0), result(0);
                    int xarray(0), yarray(0), zarray(0);
                    // points
                    float x(0), y(0), z(0);
                    // dummy
                    float dummy(0);

                    ////////////////////////////////////////////////////////
                    // The data will be parsed with help from the pattern
                    //

                    while (reading_pos1 < pattern_length)
                    {

                        // handle escape sequenzes
                        if (pattern.at(reading_pos1) == '\\')
                        {
                            switch (pattern.at(reading_pos1 + 1))
                            {
                            case 'n':
                                infile.getline(buffer, pBufferSize->getValue());
                                reading_pos1 += 2;
                                reading_pos2 = 0;
                                break;
                            default:

                                break;
                            }
                        }
                        //cout << pattern << endl;
                        // parsing the data
                        if (pattern.at(reading_pos1) == '%')
                        {
                            // parse first char
                            switch (pattern.at(reading_pos1 + 1))
                            {
                            case '%':
                                reading_pos1 += 2;
                                reading_pos2 += 2;
                                break;

                            case 'X':
                                if (pattern.at(reading_pos1 + 2) == 'P')
                                {
                                    result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                    if (result > 0)
                                        res += result;
                                    else
                                        error = true;

                                    reading_pos1 += 3;
                                    reading_pos2 += num;
                                    if (xarray < 2)
                                        para.xcoords[xarray++] = x;
                                    //cout << "found XP: " << x << ", num: " << num << endl;
                                }
                                break;

                            case 'Y':
                                if (pattern.at(reading_pos1 + 2) == 'P')
                                {
                                    result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                    if (result > 0)
                                        res += result;
                                    else
                                        error = true;

                                    reading_pos1 += 3;
                                    reading_pos2 += num;
                                    if (yarray < 2)
                                        para.ycoords[yarray++] = y;
                                    //cout << "found YP: " << y << ", num: " << num << endl;
                                }
                                break;

                            case 'Z':
                                if (pattern.at(reading_pos1 + 2) == 'P')
                                {
                                    result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                    if (result > 0)
                                        res += result;
                                    else
                                        error = true;

                                    reading_pos1 += 3;
                                    reading_pos2 += num;
                                    if (zarray < 2)
                                        para.zcoords[zarray++] = z;
                                    //cout << "found ZP: " << z << ", num: " << num << endl;
                                }
                                break;

                            case 'F':
                                result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos2 += num;
                                reading_pos1 += 2;
                                //cout << "found F: " << dummy << ", num: " << num << endl;
                                break;

                            default:
                                reading_pos1++;
                                reading_pos2++;
                                break;
                            }
                        }
                        else
                        {
                            reading_pos1++;
                            reading_pos2++;
                        }

                    } // while(reading_pos1 < pattern_length)
                }
            }
            else
            {
                while (pattern.length() > 1)
                {

                    // The pattern has to be parted into parts seperated from a \n
                    size_t pos = pattern.find("\\n", 0);
                    if (pos > 0)
                    {
                        strTemp = pattern.substr(0, pos);
                        pattern.erase(0, pos + 2);
                    }
                    else
                    {
                        strTemp = pattern;
                        pattern.erase();
                    }

                    if (infile.getline(buffer, pBufferSize->getValue()))
                    {
                        //cout << i << " - " << strTemp << "==>" << buffer << endl;

                        //helper
                        unsigned int reading_pos1(0), reading_pos2(0);
                        int num(0), res(0), result(0);
                        int xarray(0), yarray(0), zarray(0);
                        // points
                        float x(0), y(0), z(0);
                        // dummy
                        float dummy(0);

                        while (reading_pos1 < strTemp.length())
                        {
                            if (strTemp.at(reading_pos1) == '%')
                            {
                                switch (strTemp.at(reading_pos1 + 1))
                                {
                                case '%':
                                    reading_pos1 += 2;
                                    reading_pos2 += 2;
                                    break;

                                case 'X':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.xcoords[xarray++] = x;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        resCounter++;
                                        //cout << "found XP: " << x << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'Y':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.ycoords[yarray++] = y;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found YP: " << y << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'Z':
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            para.zcoords[zarray++] = z;
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        //cout << "found ZP: " << z << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                    break;

                                case 'F':
                                    result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                                    if (result > 0)
                                        res += result;
                                    else
                                        error = true;

                                    reading_pos2 += num;
                                    reading_pos1 += 2;
                                    //cout << "found F: " << dummy << ", num: " << num << endl;
                                    break;

                                default:
                                    reading_pos1++;
                                    reading_pos2++;
                                    break;

                                } // switch
                            } // if(pattern.at(reading_pos1) == '%')
                            else
                            {
                                reading_pos1++;
                                reading_pos2++;
                            }
                        }
                    } // if(infile.getline(buffer, pBufferSize->getValue()))
                } // while(pattern.length()>1)
            }

        } // if((pDimX->getValue())&&(pDimY->getValue())&&(pDimZ->getValue()))
        else
        {
            sendError("Not enough dimensions for Uniform Grid creation.");
            return false;
        }
    }
    break;

    // rectilinear grid
    case 2:
    {
        param(NULL, false);

        // read values
        string strTemp;
        int resCounter(0);

        while (pattern.length() > 0)
        {
            // The pattern has to be parted into parts seperated from a \n
            size_t pos = pattern.find("\\n", 0);
            if (pos > 0)
            {
                strTemp = pattern.substr(0, pos);
                pattern.erase(0, pos + 2);
            }
            else
            {
                strTemp = pattern;
                pattern.erase();
            }
            //helper
            unsigned int reading_pos1(0), reading_pos2(0);
            int num(0), res(0), result(0);
            // points
            float x(0), y(0), z(0);
            // dummy
            float dummy(0);

            while (reading_pos1 < strTemp.length())
            {
                if (strTemp.at(reading_pos1) == '%')
                {
                    switch (strTemp.at(reading_pos1 + 1))
                    {
                    case '%':
                        reading_pos1 += 2;
                        reading_pos2 += 2;
                        break;

                    case 'X':

                        if (strTemp.at(reading_pos1 + 2) == 'P')
                        {
                            for (int i = 0; i < pDimX->getValue(); i++)
                            {
                                infile.getline(buffer, pBufferSize->getValue());
                                reading_pos2 = reading_pos1;

                                result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);
                                if (result > 0)
                                {
                                    res += result;
                                    para.xcoords[i] = x;
                                }
                                else
                                    error = true;

                                //cout << "found XP: " << x << ", num: " << num << endl;
                            }
                            reading_pos1 += 3;
                            reading_pos2 += num;
                            resCounter++;
                        }
                        else
                        {
                            sendError("unknown reading sequence");
                            reading_pos1++;
                            reading_pos2++;
                        }
                        break;

                    case 'Y':
                        if (strTemp.at(reading_pos1 + 2) == 'P')
                        {
                            for (int i = 0; i < pDimY->getValue(); i++)
                            {
                                infile.getline(buffer, pBufferSize->getValue());
                                reading_pos2 = reading_pos1;

                                result = sscanf(buffer + reading_pos2, "%f%n", &y, &num);
                                if (result > 0)
                                {
                                    res += result;
                                    para.ycoords[i] = y;
                                }
                                else
                                    error = true;

                                //cout << "found YP: " << y << ", num: " << num << endl;
                            }
                            reading_pos1 += 3;
                            reading_pos2 += num;
                            resCounter++;
                        }
                        else
                        {
                            sendError("unknown reading sequence");
                            reading_pos1++;
                            reading_pos2++;
                        }
                        break;

                    case 'Z':
                        if (strTemp.at(reading_pos1 + 2) == 'P')
                        {
                            for (int i = 0; i < pDimZ->getValue(); i++)
                            {
                                infile.getline(buffer, pBufferSize->getValue());
                                reading_pos2 = reading_pos1;

                                result = sscanf(buffer + reading_pos2, "%f%n", &z, &num);
                                if (result > 0)
                                {
                                    res += result;
                                    para.zcoords[i] = z;
                                }
                                else
                                    error = true;

                                //cout << "found ZP: " << z << ", num: " << num << endl;
                            }
                            reading_pos1 += 3;
                            reading_pos2 += num;
                            resCounter++;
                        }
                        else
                        {
                            sendError("unknown reading sequence");
                            reading_pos1++;
                            reading_pos2++;
                        }
                        break;

                    case 'F':
                        infile.getline(buffer, pBufferSize->getValue());
                        result = sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);
                        reading_pos1 += 3;
                        reading_pos2 += num;
                        break;

                    default:
                        reading_pos1++;
                        reading_pos2++;
                        break;

                    } // switch
                } // if(pattern.at(reading_pos1) == '%')
                else
                {
                    reading_pos1++;
                    reading_pos2++;
                }
            } // while(reading_pos1 < strTemp.length())

        } // while(pattern.length() > 0)
    }
    break;

    // structured grid
    case 3:
    {

        // if vectors not in std sequence, we have to sort them. First make an array with the right permutation
        int *perm = new int[num_of_points];

        if (pCoordSequence->getValue() == 0)
        {
            for (int i = 0; i < num_of_points; i++)
            {
                int count1 = (i * dimY * dimZ) % (dimX * dimY * dimZ);
                int count2 = (int)(i / dimX);
                int modulo = dimX * dimY * dimZ;

                perm[i] = ((count1 + (int)(count2 / dimY) + ((dimZ * count2) % (dimY * dimZ))) % modulo);
                //cout << i << " --> " << perm << endl;
            }
        }

        // read data
        int arrayCounter = 0;
        int resCounter = 0;

        string strTemp;
        // data interleaving or not
        if (pInterl->getValue())
        {
            while ((infile.getline(buffer, pBufferSize->getValue())))
            {

                //helper
                size_t pattern_length(pattern.length());
                unsigned int reading_pos1(0), reading_pos2(0);
                int num(0), res(0), result(0);
                // points
                float x(0), y(0), z(0);
                // dummy
                float dummy(0);

                ////////////////////////////////////////////////////////
                // The data will be parsed with help from the pattern
                //

                while (reading_pos1 < pattern_length)
                {

                    // handle escape sequenzes
                    if (pattern.at(reading_pos1) == '\\')
                    {
                        switch (pattern.at(reading_pos1 + 1))
                        {
                        case 'n':
                            infile.getline(buffer, pBufferSize->getValue());
                            reading_pos1 += 2;
                            reading_pos2 = 0;
                            break;
                        default:

                            break;
                        }
                    }

                    // parsing the data
                    if (pattern.at(reading_pos1) == '%')
                    {
                        // parse first char
                        switch (pattern.at(reading_pos1 + 1))
                        {
                        case '%':
                            reading_pos1 += 2;
                            reading_pos2 += 2;
                            break;

                        case 'X':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                if (result > 0)
                                {
                                    res += result;
                                    if (pCoordSequence->getValue() != 0)
                                        para.xcoords[arrayCounter] = x;
                                    else
                                        para.xcoords[perm[arrayCounter]] = x;
                                }
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found XP: " << x << ", num: " << num << endl;
                            }
                            break;

                        case 'Y':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                if (result > 0)
                                {
                                    res += result;
                                    if (pCoordSequence->getValue() != 0)
                                        para.ycoords[arrayCounter] = y;
                                    else
                                        para.ycoords[perm[arrayCounter]] = y;
                                }
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found YP: " << y << ", num: " << num << endl;
                            }
                            break;

                        case 'Z':
                            if (pattern.at(reading_pos1 + 2) == 'P')
                            {
                                result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                if (result > 0)
                                {
                                    res += result;
                                    if (pCoordSequence->getValue() != 0)
                                        para.zcoords[arrayCounter] = z;
                                    else
                                        para.zcoords[perm[arrayCounter]] = z;
                                }
                                else
                                    error = true;

                                reading_pos1 += 3;
                                reading_pos2 += num;
                                //cout << "found ZP: " << z << ", num: " << num << endl;
                            }
                            break;

                        case 'F':
                            result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                            if (result > 0)
                                res += result;
                            else
                                error = true;

                            reading_pos2 += num;
                            reading_pos1 += 2;
                            //cout << "found F: " << dummy << ", num: " << num << endl;
                            break;

                        default:
                            reading_pos1++;
                            reading_pos2++;
                            break;
                        }
                    }
                    else
                    {
                        reading_pos1++;
                        reading_pos2++;
                    }

                } // while(reading_pos1 < pattern_length)

                if ((res > 0) && (arrayCounter < num_of_points))
                {
                    resCounter++;
                    arrayCounter++;
                }
            } // while((infile.getline(buffer, pBufferSize->getValue())))

            delete[] perm;

        } // if(!pInterl->getValue())
        else
        {

            while (pattern.length() > 1)
            {

                // The pattern has to be parted into parts seperated from a \n
                size_t pos = pattern.find("\\n", 0);
                if (pos > 0)
                {
                    strTemp = pattern.substr(0, pos);
                    pattern.erase(0, pos + 2);
                }
                else
                {
                    strTemp = pattern;
                    pattern.erase();
                }

                int xCount(0), yCount(0), zCount(0);

                // iteration over the points to read
                while ((xCount < num_of_points) && (yCount < num_of_points) && (zCount < num_of_points) && (infile.getline(buffer, pBufferSize->getValue())))
                {

                    //helper
                    unsigned int reading_pos1(0), reading_pos2(0);
                    int num(0), res(0), result(0);
                    // points
                    float x(0), y(0), z(0);
                    // dummy
                    float dummy(0);

                    while (reading_pos1 < strTemp.length())
                    {
                        if (strTemp.at(reading_pos1) == '%')
                        {
                            switch (strTemp.at(reading_pos1 + 1))
                            {
                            case '%':
                                reading_pos1 += 2;
                                reading_pos2 += 2;
                                break;

                            case 'X':

                                if (strTemp.length() >= reading_pos1 + 2)
                                {
                                    if (strTemp.at(reading_pos1 + 2) == 'P')
                                    {
                                        result = sscanf(buffer + reading_pos2, "%f%n", &x, &num);

                                        if (result > 0)
                                        {
                                            res += result;
                                            if (xCount < num_of_points)
                                            {
                                                if (pCoordSequence->getValue() != 0)
                                                    para.xcoords[xCount] = x;
                                                else
                                                    para.xcoords[((xCount * dimY * dimZ) + (int)(xCount / dimX)) % (dimX * dimY * dimZ)] = x;

                                                xCount++;
                                            }
                                        }
                                        else
                                            error = true;

                                        reading_pos1 += 3;
                                        reading_pos2 += num;
                                        resCounter++;
                                        //cout << "found XP: " << x << ", num: " << num << endl;
                                    }
                                    else
                                    {
                                        sendError("unknown reading sequence");
                                        reading_pos1++;
                                        reading_pos2++;
                                    }
                                }
                                break;

                            case 'Y':
                                if (strTemp.at(reading_pos1 + 2) == 'P')
                                {
                                    result += sscanf(buffer + reading_pos2, "%f%n", &y, &num);

                                    if (result > 0)
                                    {
                                        res += result;
                                        if (yCount < num_of_points)
                                        {
                                            if (pCoordSequence->getValue() != 0)
                                                para.ycoords[yCount++] = y;
                                            else
                                                para.ycoords[yCount++] = y;
                                        }
                                    }
                                    else
                                        error = true;

                                    reading_pos1 += 3;
                                    reading_pos2 += num;
                                    //cout << "found YP: " << y << ", num: " << num << endl;
                                }
                                else
                                {
                                    sendError("unknown reading sequence");
                                    reading_pos1++;
                                    reading_pos2++;
                                }
                                break;

                            case 'Z':
                                if (strTemp.at(reading_pos1 + 2) == 'P')
                                {
                                    result += sscanf(buffer + reading_pos2, "%f%n", &z, &num);

                                    if (result > 0)
                                    {
                                        res += result;

                                        if (pCoordSequence->getValue() != 0)
                                            para.zcoords[yCount++] = z;
                                        else
                                            para.zcoords[(((zCount * dimZ) + (int)(zCount / (dimX * dimY))) % (dimX * dimY * dimZ))] = z;
                                        zCount++;
                                    }
                                    else
                                        error = true;

                                    reading_pos1 += 3;
                                    reading_pos2 += num;
                                    //cout << "found ZP(" << zCount-1 << "): " << z << ", num: " << num << endl;
                                }
                                else
                                {
                                    sendError("unknown reading sequence");
                                    reading_pos1++;
                                    reading_pos2++;
                                }
                                break;

                            case 'F':
                                result += sscanf(buffer + reading_pos2, "%f%n", &dummy, &num);

                                if (result > 0)
                                    res += result;
                                else
                                    error = true;

                                reading_pos2 += num;
                                reading_pos1 += 2;
                                //cout << "found F: " << dummy << ", num: " << num << endl;
                                break;

                            default:
                                reading_pos1++;
                                reading_pos2++;
                                break;

                            } // switch
                        } // if(pattern.at(reading_pos1) == '%')
                        else
                        {
                            reading_pos1++;
                            reading_pos2++;
                        }
                    }
                } // while(infile.getline(buffer, pBufferSize->getValue())
            } // while(pattern.length()>1)
        } // else
    }
    break;
    }

    // clean up
    infile.close();
    delete[] buffer;

    return true;
}

void ReadASCII::param(const char * /*name*/, bool /*inMapLoading*/)
{
    // map parameter to control panel in dependece to the choosen geometry
    //

    // some Parameter are always mapped
    pbrFile->show();
    pBufferSize->show();
    pGeom->show();
    pDataByteOffset->show();
    pDataLineOffset->show();
    pDataFormat->show();
    pOutputResult->show();
    pOutputDebug->show();

    // points
    if (pGeom->getValue() == 0)
    {
        pHeader->hide();
        pHeaderByteOffset->hide();
        pHeaderLineSkip->hide();
        pDimPattern->hide();
        pDimX->hide();
        pUniDistX->hide();
        pDimY->hide();
        pUniDistY->hide();
        pDimZ->hide();
        pUniDistZ->hide();
        pInterl->show();
        pPointsNum->show();
        pCoordSequence->hide();
    }

    // uniform grid
    if (pGeom->getValue() == 1)
    {
        pDimX->show();
        pUniDistX->hide();
        pDimY->show();
        pUniDistY->hide();
        pDimZ->show();
        pUniDistZ->hide();
        pInterl->show();
        pPointsNum->hide();
        pCoordSequence->hide();
        pHeader->show();
        if (pHeader->getValue())
        {
            pHeaderByteOffset->show();
            pHeaderLineSkip->show();
            pDimPattern->show();
        }
        else
        {
            pHeaderByteOffset->hide();
            pHeaderLineSkip->hide();
            pDimPattern->hide();
        }
    }
    // rectilinear grid
    if (pGeom->getValue() == 2)
    {
        pDimX->show();
        pUniDistX->hide();
        pDimY->show();
        pUniDistY->hide();
        pDimZ->show();
        pUniDistZ->hide();
        pInterl->hide();
        pPointsNum->hide();
        pCoordSequence->hide();
        pHeader->show();
        if (pHeader->getValue())
        {
            pHeaderByteOffset->show();
            pHeaderLineSkip->show();
            pDimPattern->show();
        }
        else
        {
            pHeaderByteOffset->hide();
            pHeaderLineSkip->hide();
            pDimPattern->hide();
        }
    }

    // structured grid
    if (pGeom->getValue() == 3)
    {
        pDimX->show();
        pUniDistX->show();
        pDimY->show();
        pUniDistY->show();
        pDimZ->show();
        pUniDistZ->show();
        pInterl->show();
        pPointsNum->hide();
        pCoordSequence->show();
        pHeader->show();
        if (pHeader->getValue())
        {
            pHeaderByteOffset->show();
            pHeaderLineSkip->show();
            pDimPattern->show();
        }
        else
        {
            pHeaderByteOffset->hide();
            pHeaderLineSkip->hide();
            pDimPattern->hide();
        }
    }

    // compute header
    if (pHeader->getValue())
    {
        // deklarations
        const char *path = pbrFile->getValue();
        ifstream infile(path, ios::in);
        int offsetCounter(0);
        int res(0);
        int dimX(0), dimY(0), dimZ(0);
        char *buffer = new char[pBufferSize->getValue()];

        // is filestream correct?
        if (!infile)
        {
            sendError("Checkpoint file %s not found.", path);
        }

        // skip offsets
        if (pHeaderLineSkip->getValue() > 0)
            while ((infile.getline(buffer, pBufferSize->getValue()) && (offsetCounter < pHeaderLineSkip->getValue() - 1)))
                offsetCounter++;
        if (pHeaderByteOffset->getValue() > 0)
            infile.seekg(pHeaderByteOffset->getValue(), ios::beg);

        // read the dimensions from header
        if (pGeom->getValue() != 0)
        {
            infile.getline(buffer, pBufferSize->getValue());

            res = sscanf(buffer, pDimPattern->getValue(), &dimX, &dimY, &dimZ);
        }

        // store dimensions into class
        if (res == 3)
        {
            pDimX->setValue(dimX);
            pDimY->setValue(dimY);
            pDimZ->setValue(dimZ);

            if (pGeom->getValue() == 3)
                pPointsNum->setValue(pDimX->getValue() * pDimY->getValue() * pDimZ->getValue());
        }
        else
            printf("Too less dimensions found.Check dim pattern and header offset/line skip.");

        // output results
        if (pOutputResult->getValue())
        {
            cout << "\n===Header===\n";
            printf("Dim Pattern:\t\t%s\n", pDimPattern->getValue());
            cout << "Dim line from file:\t" << buffer << endl;
            printf("# Dimensions found:\t%d\n", res);
            printf("Dimensions are:\t\t%d, %d, %d\n", dimX, dimY, dimZ);
        }

        //clean up
        infile.close();
        delete[] buffer;
    }

    /*
     if(paramName == "")
     sendError("ein Fehler ist aufgetreten\n");
     if(paramName == pGeom->getName());
     if(paramName == pSedPattern->getName());
     if(paramName == pHeader->getName());
     if(paramName == pHeaderByteOffset->getName());
     if(paramName == pHeaderLineSkip->getName());
     if(paramName == pDimPattern->getName());
     if(paramName == pDimX->getName());
     if(paramName == pDimY->getName());
     if(paramName == pDimZ->getName());
     if(paramName == pInterl->getName());
     if(paramName == pPointsNum->getName());
     if(paramName == pDataByteOffset->getName());
     if(paramName == pDataLineOffset->getName());
     if(paramName == pDataFormat->getName());
   */
}

ReadASCII::~ReadASCII()
{
}

void ReadASCII::printResult(params para)
{
    cout << "\n===Data===\n";

    if (para.xcoords != NULL)
        cout << " coords";
    if (para.scalardata != NULL)
        cout << " \t scalar_data";
    if (para.xdata != NULL)
        cout << "  vector_data";
    cout << endl << "------------------------------------\n";

    if (pGeom->getValue() != 2)
    {
        for (int i = 0; i < pPointsNum->getValue(); i++)
        {

            if (para.xcoords != NULL)
                cout << " " << para.xcoords[i] << ",\t";
            if (para.ycoords != NULL)
                cout << para.ycoords[i] << ",\t";
            if (para.zcoords != NULL)
                cout << para.zcoords[i] << "   ";
            if (para.scalardata != NULL)
                cout << " | " << para.scalardata[i] << "   ";
            if (para.xdata != NULL)
                cout << " | " << para.xdata[i] << ", ";
            if (para.ydata != NULL)
                cout << para.ydata[i] << ", ";
            if (para.zdata != NULL)
                cout << para.zdata[i];
            cout << endl;
        }
    }
    else
    {
        cout << "\nx_coords: ";
        for (int i = 0; i < pDimX->getValue(); i++)
            cout << para.xcoords[i] << "\t";

        cout << "\ny_coords: ";
        for (int i = 0; i < pDimY->getValue(); i++)
            cout << para.ycoords[i] << " \t";

        cout << "\nz_coords: ";
        for (int i = 0; i < pDimZ->getValue(); i++)
            cout << para.zcoords[i] << "\t";
        cout << endl;
    }
}

void ReadASCII::printDebug(void)
{
    string geom;
    if (pGeom->getValue() == 0)
        geom = "Points";
    if (pGeom->getValue() == 1)
        geom = "Uniform Grid";
    if (pGeom->getValue() == 2)
        geom = "Rectilinear Grid";
    if (pGeom->getValue() == 3)
        geom = "Structured Grid";

    cout << "\n\n==================\nDebug Information\n";
    cout << "Parameters in use:\n==================\n"
         << "File Path\t\t" << pbrFile->getValue() << endl
         << "Max line length\t\t" << pBufferSize->getValue() << endl
         << "Geometry\t\t" << geom << endl
         << "Data Byte Offset\t" << pDataByteOffset->getValue() << endl
         << "Data line Skip\t\t" << pDataLineOffset->getValue() << endl
         << "Data format\t\t" << pDataFormat->getValue() << endl << endl;

    // points
    if (pGeom->getValue() == 0)
    {
        cout
            << "Interleaved Format\t" << pInterl->getValue() << endl
            << "Number of points\t" << pPointsNum->getValue() << endl << endl;
    }

    // uniform grid
    if (pGeom->getValue() == 1)
    {
        cout << "Header\t\t\t" << pHeader->getValue() << endl;
        if (pHeader->getValue())
        {
            cout
                << "Header Byte Offset\t" << pHeaderByteOffset->getValue() << endl
                << "Header Line Offset\t" << pHeaderLineSkip->getValue() << endl
                << "Dimension Pattern\t" << pDimPattern->getValue() << endl;
        }
        cout
            << "X Dimension\t\t" << pDimX->getValue() << endl
            << "Y Dimension\t\t" << pDimY->getValue() << endl
            << "Z Dimension\t\t" << pDimZ->getValue() << endl
            << "Interleaved Format\t" << pInterl->getValue() << endl << endl;
    }
    // rectilinear grid
    if (pGeom->getValue() == 2)
    {
        cout << "Header\t\t\t" << pHeader->getValue() << endl;
        if (pHeader->getValue())
        {
            cout
                << "Header Byte Offset\t" << pHeaderByteOffset->getValue() << endl
                << "Header Line Offset\t" << pHeaderLineSkip->getValue() << endl
                << "Dimension Pattern\t" << pDimPattern->getValue() << endl;
        }
        cout
            << "X Dimension\t\t" << pDimX->getValue() << endl
            << "Y Dimension\t\t" << pDimY->getValue() << endl
            << "Z Dimension\t\t" << pDimZ->getValue() << endl << endl;
    }

    // structured grid
    if (pGeom->getValue() == 3)
    {
        string gis;
        if (pCoordSequence->getValue() == 0)
            gis = "i, j, k";
        if (pCoordSequence->getValue() == 1)
            gis = "k, j, i";

        cout << "Header\t\t\t" << pHeader->getValue() << endl;
        if (pHeader->getValue())
        {
            cout
                << "Header Byte Offset\t" << pHeaderByteOffset->getValue() << endl
                << "Header Line Offset\t" << pHeaderLineSkip->getValue() << endl
                << "Dimension Pattern\t" << pDimPattern->getValue() << endl;
        }
        cout
            << "X Dimension\t\t" << pDimX->getValue() << endl
            << "X Uniform Distance\t" << pUniDistX->getValue() << endl
            << "Y Dimension\t\t" << pDimY->getValue() << endl
            << "Y Uniform Distance\t" << pUniDistY->getValue() << endl
            << "Z Dimension\t\t" << pDimZ->getValue() << endl
            << "Z Uniform Distance\t" << pUniDistZ->getValue() << endl
            << "Interleaved Format\t" << pInterl->getValue() << endl
            << "Grid Index Sequence\t" << gis << endl << endl;
    }
}

bool strContains(const string &str1, const string &str2)
{

    if ((str1.find(str2, 0) > 0) && !(str1.find(str2, 0) > str1.length()))
    {

        return true;
    }
    if (str1.substr(0, str2.length()) == str2)
    {

        return true;
    }

    return false;
}

void expandPattern(string &pattern, int bufferSize)
{
    size_t pos(pattern.find("%XP("));
    string output(pattern);
    char *del = new char[bufferSize];
    if (((pos > 0) && !(pos > output.length())) || (output.substr(0, 4) == "%XP("))
    {
        int num1(0), num2(0);

        size_t pos2(output.find(")", pos + 4));
		size_t pos3(output.find(",", pos + 4));

        pos += 4;
        if (sscanf(output.substr(pos, pos2 - pos).c_str(), "%d%n", &num1, &num2) < 1)
        {
            fprintf(stderr, "ReadASCII: expandPattern: sscanf failed\n");
        }

        string strDel(output.substr(pos3 + 1, pos2 - pos3 - 1));

        string strTemp1("%XP");

        string strTemp2;
        for (int i = 0; i < num1 - 1; i++)
            strTemp2 += strTemp1 + strDel;
        strTemp2 += strTemp1;

        output.replace(pos - 4, pos2 - pos + num2 + 4, strTemp2);
    }
    //cout << "new Pattern: " << output << endl;

    pos = output.find("%YP(");
    if (((pos > 0) && !(pos > output.length())) || (output.substr(0, 4) == "%YP("))
    {
        int num1(0), num2(0);

        size_t pos2(output.find(")", pos + 4));
        size_t pos3(output.find(",", pos + 4));

        pos += 4;
        if (sscanf(output.substr(pos, pos2 - pos).c_str(), "%d%n", &num1, &num2) < 1)
        {
            fprintf(stderr, "ReadASCII: expandPattern: sscanf2 failed\n");
        }

        string strDel(output.substr(pos3 + 1, pos2 - pos3 - 1));

        string strTemp1("%YP");

        string strTemp2;
        for (int i = 0; i < num1 - 1; i++)
            strTemp2 += strTemp1 + strDel;
        strTemp2 += strTemp1;

        output.replace(pos - 4, pos2 - pos + num2 + 4, strTemp2);
    }
    //cout << "new Pattern: " << output << endl;

    pos = output.find("%ZP(");
    if (((pos > 0) && !(pos > output.length())) || (output.substr(0, 4) == "%ZP("))
    {
        int num1(0), num2(0);

        size_t pos2(output.find(")", pos + 4));
        size_t pos3(output.find(",", pos + 4));

        pos += 4;
        if (sscanf(output.substr(pos, pos2 - pos).c_str(), "%d%n", &num1, &num2) < 1)
        {
            fprintf(stderr, "ReadASCII: expandPattern: sscanf3 failed\n");
        }

        string strDel(output.substr(pos3 + 1, pos2 - pos3 - 1));

        string strTemp1("%ZP");

        string strTemp2;
        for (int i = 0; i < num1 - 1; i++)
            strTemp2 += strTemp1 + strDel;
        strTemp2 += strTemp1;
        output.replace(pos - 4, pos2 - pos + num2 + 4, strTemp2);
    }

    //cout << "new Pattern: " << output << endl;
    pattern = output;
    delete[] del;
}

void hello(const string str)
{
    cout << "hello " << str << endl;
}

void hello(const int str)
{
    cout << "hello " << str << endl;
}

MODULE_MAIN(Reader, ReadASCII)
