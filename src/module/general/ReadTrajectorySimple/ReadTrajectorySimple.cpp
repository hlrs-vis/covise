
//lenght of a line - is this the line size for the file I am reading from? 
//in my case would it be the number of time stamps per trajectory?
#define LINE_SIZE 8192

// portion for resizing data
//?used as the initial size and increment for dynamic arrays.
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadTrajectorySimple.h"
#include <do/coDoLines.h>//coDoLines

#include <iostream>
#include <vector>
#include <limits>
using namespace std;

//Constructor for ReadTrajectory Object
//what is argc and array argv used for? 
//its not used inside the constructor and in the manual there is no indication of 
//what it is used for within the coModule constructor either


ReadTrajectorySimple::ReadTrajectorySimple(int argc, char *argv[])
    : coModule(argc, argv, "Simple Trajectory Reader")
{
    // Returns a pointer to the output port
    linePort = addOutputPort("lines", "Lines", "geometry lines");

    // select the OBJ file name with a file browser
    //A File Browser parameter allows the selection of a file on the host the module is running on.
    //addFileBrowserParam: Creates a browser parameter
    //objFileParam: pointer to newly created port
    objFileParam = addFileBrowserParam("crtfFile", "crtf file"); //where is obj file? this should be my test data?
    //The start value of a browser parameter must be set by the module in the constructor
    //IN: default file name with path, file selection mask, e.g. "*.dat"
    objFileParam->setValue("/data/CapeReviso/visagx-testdata/vhs/medium_playback1_2023-01-13_11-52-31.crtf", "*.crtf");
    //setValue returns =0 on error, =1 on success so why set objFileParam to 1/0?
}

ReadTrajectorySimple::~ReadTrajectorySimple()//No destructor?
{
}

void ReadTrajectorySimple::quit()//does not return?
{
}

int ReadTrajectorySimple::compute(const char *port)
{
    /*(void) is a type cast that is used to explicitly indicate to the compiler that the 
    variable port is intentionally not used within the compute function.*/
    (void)port;

    // get the file name
    filename = objFileParam->getValue();

    if (filename != NULL)
    {
        // open the file
        if (openFile())
        {
            sendInfo("File %s open", filename);

            // read the file, create the lists and create a COVISE line object
            readFile();
        }
        else
        {
            sendError("Error opening file %s", filename);
            return FAIL;
        }
    }
    else
    {
        sendError("ERROR: fileName is NULL");
        return FAIL;
    }
    return SUCCESS;
}


void
ReadTrajectorySimple::readFile()
{
    int numCoords = 0, numTimeStamps = 0, numCorners = 0, numLines = 0;

    char line[LINE_SIZE]; // line in an obj file
    int trajectoryNumber, typeCount, totalTimeStamps;
    char trajectoryType[100]; // Make sure the char array is large enough to hold the string
    float relativeTimeZero = numeric_limits<float>::max(), startTimeStamp, lastTimeStamp;
    float xPosition, yPosition, TimeStamp;
    int numScanned, numCoordScanned; // number of characters scanned with sscanf
    float zCoord = 0;
    float xValue, yValue, timeValue;
    coDoLines *lineObject; // output object
    const char *lineObjectName; // output object name assigned by the controller

    //Declaring Vectors for x_coordinates, y_coordinates, z_coordinates, corner_list, line_list
    vector<float> x_coordinates;
    vector<float> y_coordinates;
    vector<float> z_coordinates;
    vector<float> timeStamp_list;
    vector<int> corner_list;
    vector<int> line_list;
    
    // read one line after another
    while (fgets(line, LINE_SIZE, fp) != NULL)
    {
        // read the keyword
        numScanned = sscanf(line, "%d %d %99s %d %f %f",
                                 &trajectoryNumber, &typeCount, trajectoryType,
                                 &totalTimeStamps, &startTimeStamp, &lastTimeStamp);
        //set the relative time stamp
        if (startTimeStamp > relativeTimeZero)
        {
            relativeTimeZero = startTimeStamp;
        }
        //test if it is header line or timestamp w/ coordinates
        if ( numScanned != 6)
        {
            numScanned = sscanf(line, " %f %f %f",
                                 &xPosition, &yPosition, &TimeStamp);

            if (numScanned != 3){
                cerr << "ReadTrajectorySimple::readFile:: sscanf1 failed" << endl;
            }
        }
        /*2 cases
        1. numScanned = 6 --> Header Line --> start of a new trajectory
            num_l
        2. numScanned = 3 --> Points in the trajectory*/
        if (numScanned == 6)
        {
            /*The variable numLines is incremented
            indicating that a new line is encountered.*/
            numLines++;
            /*What do we do about the type of trajectory?
            if (strcasecmp("car", trajectoryType) == 0){

            }
            if (strcasecmp("person", trajectoryType) == 0){

            }
            if (strcasecmp("bicycle", trajectoryType) == 0){

            }
            */
            line_list.push_back(numCorners);
        } else if (numScanned == 3) 
        {
            /*Line with point coord + timestamp*/
            numCoordScanned = sscanf(line, "%f %f %f", &xValue, &yValue, &timeValue);
            if (numCoordScanned != 3)
            {
                cerr << "Failed to read two floats from the line." << endl;
            }
            numCoords++;
            numCorners++;
            numTimeStamps++;
            /*Read the coordinate value and store in x and y, store 0 in z*/
            x_coordinates.push_back(xValue);
            y_coordinates.push_back(yValue);
            z_coordinates.push_back(zCoord);
            timeStamp_list.push_back(relativeTimeZero - startTimeStamp);
            // Append numCoord to corner list
            corner_list.push_back(numCoords - 1);
        } else
        {
            cerr << "ReadTrajectorySimple::readFile:: sscanf failed" << endl;
        }
    }

    sendInfo("found %d coordinates, %d corners, %d lines", numCoords, numCorners, numLines);

    // get the COVISE output object name from the controller
    lineObjectName = linePort->getObjName();

    // create the COVISE output object
    /*Creating pointer to the first element in the vector
    Do not need to dealocate memory since no new array is created and memory
    allocated for the vector will be automatically deallocated by its destructor*/
    float* cx = x_coordinates.data();
    float* cy = y_coordinates.data();
    float* cz = z_coordinates.data();
    int*   ci = corner_list.data();
    int*   li = line_list.data();
    
    lineObject = new coDoLines(lineObjectName, numCoords, cx, cy, cz, numCorners, ci, numLines, li);
    linePort->setCurrentObject(lineObject);
   
}

bool ReadTrajectorySimple::openFile()
{
    sendInfo("Opening file %s", filename);

    // open the obj file
    if ((fp = Covise::fopen((char *)filename, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", filename);
        return false;
    }
    else
    {
        return true;
    }
}



MODULE_MAIN(IO, ReadTrajectorySimple)
