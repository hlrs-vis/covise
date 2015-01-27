/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///////////////////////////////////////////////////////////////
// Module:  			ReadWMatrix.cpp
// Author:  			Juergen Schulze-Doebold
// Creation date: 6/16/1999
// Function: 			Reads a world lines matrix file and outputs volume data
// Relationship: 	SFB 382, Project A8
// Usage:					Provide filename of matrix file
///////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include "ReadWMatrix.h"

// Values which produce warnings when overflow:
#define MAX_X 1000.0
#define MAX_Y 1000.0
#define MAX_Z 1000.0
#define MAX_COL 2.0

int main(int argc, char *argv[])
{
    ReadWMatrix *application = new ReadWMatrix(argc, argv);
    application->run();

    return 0;
}

ReadWMatrix::ReadWMatrix(int argc, char *argv[])
{
    // this info appears in the module setup window
    Covise::set_module_description("WorldLinesMatrix Reader");

    // the output ports
    Covise::add_port(OUTPUT_PORT, "matrix", "coDoUniformGrid", "Matrix Coordinates");
    Covise::add_port(OUTPUT_PORT, "matrixColors", "coDoFloat", "Colors of Matrix Vertices");

    // select the OBJ file name with a file browser
    Covise::add_port(PARIN, "wmPath", "Browser", "WorldLineMatrix File");
    Covise::set_port_default("wmPath", "/mnt/cod/worldlines/ *.wm");

    // the selected lines should be highlighted
    Covise::add_port(PARIN, "scaleY", "Scalar", "Scale value in Y direction");
    Covise::set_port_default("scaleY", "-1");

    // set up the connection to the controller and data manager
    Covise::init(argc, argv);

    // set the quit and the compute callback
    Covise::set_quit_callback(ReadWMatrix::quitCallback, this);
    Covise::set_start_callback(ReadWMatrix::computeCallback, this);
}

void ReadWMatrix::quitCallback(void *userData, void *callbackData)
{
    ReadWMatrix *thisApp = (ReadWMatrix *)userData;
    thisApp->quit(callbackData);
}

void ReadWMatrix::computeCallback(void *userData, void *callbackData)
{
    ReadWMatrix *thisApp = (ReadWMatrix *)userData;
    thisApp->compute(callbackData);
}

void ReadWMatrix::quit(void *)
{
    // dummy
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// returns 1 if ok, -1 if error
// read word can be found in buf
int ReadWMatrix::getWord(char *buf)
{
    int i;
    char c;

    i = 0;
    do
    {
        do
        {
            c = fgetc(fp);
        } while (i == 0 && !feof(fp) && (c == ' ' || c == '\n' || c == '\r'));
        buf[i] = c;
        ++i;
    } while (c != ' ' && c != '\n' && c != '\r' && !feof(fp));
    buf[i] = '\0';
    if (feof(fp))
        return -1;
    else
        return 1;
}

// Reads the next float number from a text file
// returns 1 if ok, -1 if error
int ReadWMatrix::getNumber(float *number)
{
    char buffer[80];

    if (getWord(buffer) == -1)
        return -1;

    *number = (float)atof(buffer);

    //	cerr << "Read number: " << *number << endl;
    return 1;
}

// returns 1 if ok, otherwise -1
int ReadWMatrix::readWMFile()
{
    int x, y, z; // current world matrix coordinates
    int numPoints; // number of matrix points
    int error; // error number for function call
    int i; // index
    float fbuf; // buffer

    cerr << "Reading matrix source file..." << endl;

    // Read matrix dimensions from file:
    error = getNumber(&fbuf);
    if (error != 1)
        return -1;
    sizeX = (int)fbuf;
    error = getNumber(&fbuf);
    if (error != 1)
        return -1;
    sizeY = (int)fbuf;
    error = getNumber(&fbuf);
    if (error != 1)
        return -1;
    sizeZ = (int)fbuf;

    // Allocate space for matrix points:
    numPoints = sizeX * sizeY * sizeZ;
    if (numPoints > 10000000L) // break on too many points
    {
        cout << "Too many points." << endl;
        return -1;
    }
    matrixPoints = (float *)new float[numPoints];

    // Read matrix points from file:
    i = 0;
    for (z = 0; z < sizeZ; ++z)
        for (y = 0; y < sizeY; ++y)
            for (x = 0; x < sizeX; ++x)
            {
                error = getNumber(&matrixPoints[i]);
                if (error == 1)
                    ++i;
                else
                    break;
            }

    cerr << i << " matrix points in file." << endl;
    cerr << numPoints << " matrix points expected." << endl;
    if (i != numPoints)
        return -1;
    else
        return 1;
}

void ReadWMatrix::generateMatrix(float scaleY)
{
    coDoUniformGrid *unigrid = NULL;
    coDoFloat *colors = NULL;
    char *coviseObjName = NULL;
    int i, x, y, z;
    float *col;

    if (scaleY < 0.0)
        scaleY = 1.0;

    if (sizeX < 1 || sizeY < 1 || sizeZ < 1)
        return;

    // Covise-Objekte anfordern:
    coviseObjName = Covise::get_object_name("matrix");
    unigrid = new coDoUniformGrid(coviseObjName, sizeX, sizeZ, sizeY,
                                  0.0, (float)(sizeX - 1),
                                  0.0, scaleY * (float)(sizeZ - 1),
                                  0.0, -1.0 * (float)(sizeY - 1));
    coviseObjName = Covise::get_object_name("matrixColors");
    colors = new coDoFloat(coviseObjName, sizeX * sizeY * sizeZ);

    // Daten-Pointer von Covise anfordern:
    colors->getAddress(&col);
    if (colors == NULL)
        return;

    // Enter matrix points into covise data structure:
    i = 0;
    for (x = 0; x < sizeX; ++x)
        for (y = 0; y < sizeY; ++y)
            for (z = 0; z < sizeZ; ++z)
            {
                col[i] = matrixPoints[x + y * sizeX + z * sizeX * sizeY];
                ++i;
            }

    // Zugriffe auf Covise-Objekte aufraeumen:
    delete colors;
    delete unigrid;
}

void ReadWMatrix::compute(void *)
{
    float scaleY;
    char *filename;
    int error;

    // get parameters from covise
    Covise::get_scalar_param("scaleY", &scaleY);
    if (scaleY < 0.0)
        scaleY = 1.0;
    Covise::get_browser_param("wmPath", &filename);
    fp = Covise::fopen(filename, "r");
    if (!fp)
    {
        Covise::sendError("Could not open file");
        return;
    }

    // Initialize matrix parameters:
    sizeX = sizeY = sizeZ = 0;
    matrixPoints = NULL;

    // Read matrix file:
    error = readWMFile();
    fclose(fp);
    if (error != 1)
        return; // break on error

    // Convert read matrix data to covise data type:
    cerr << "Generating matrix..." << endl;
    generateMatrix(scaleY); // generate world matrix from data array

    cerr << "Freeing memory..." << endl;
    freeWMMemory();

    cerr << "Done." << endl;
    return;
}

// this method gets rid of the allocated memory for the world matrix
void ReadWMatrix::freeWMMemory()
{
    // Delete world matrix datasets:
    delete[] matrixPoints;
    sizeX = sizeY = sizeZ = 0;
}
