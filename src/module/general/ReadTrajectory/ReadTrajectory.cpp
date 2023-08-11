
//lenght of a line - is this the line size for the file I am reading from? 
//in my case would it be the number of time stamps per trajectory?
#define LINE_SIZE 8192

// portion for resizing data
//?used as the initial size and increment for dynamic arrays.
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadTrajectory.h"
#include <do/coDoLines.h>//coDoLines

#include <iostream>
#include <vector>
#include <limits>
using namespace std;

//Constructor for ReadTrajectory Object
//what is argc and array argv used for? 
//its not used inside the constructor and in the manual there is no indication of 
//what it is used for within the coModule constructor either

//point constructor
Point::Point(float x_c, float y_c, float z_c, float time_c, int corner_list_index, int line_list_index)
    : x(x_c), y(y_c), z(z_c), time(time_c), cornerIndex(corner_list_index), lineIndex(line_list_index) {}

float Point::getX() const { return x; }
float Point::getY() const { return y; }
float Point::getZ() const { return z; }
float Point::getTime() const { return time; }
int   Point::getCorner() const{ return cornerIndex; }
int   Point::getLine() const {return lineIndex; }


bool withinRadius(const Point& p1, const Point& p2, float radius) 
{
    float distanceSquared = pow(p1.getX() - p2.getX(), 2) +
                             pow(p1.getY() - p2.getY(), 2) +
                             pow(p1.getZ() - p2.getZ(), 2);
    return (distanceSquared <= pow(radius, 2));
}

vector<pair<size_t, size_t>> findPointsWithinRadius(const vector<Point>& points, float radius, float startTime, float endTime) {
    vector<pair<size_t, size_t>> intersectingIndices;

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            if (points[i].getTime() >= startTime && points[i].getTime() <= endTime &&
                points[j].getTime() >= startTime && points[j].getTime() <= endTime &&
                withinRadius(points[i], points[j], radius)) {
                intersectingIndices.push_back(make_pair(i, j));
            }
        }
    }

    return intersectingIndices;
}

vector<int> getElementsInRange(const vector<int>& inputVector, size_t startIndex, size_t endIndex) {
    vector<int> outputVector;

    if (startIndex < endIndex && endIndex <= inputVector.size()) {
        for (size_t i = startIndex; i < endIndex; ++i) {
            outputVector.push_back(inputVector[i]);
        }
    } else {
        cerr << "Invalid start or end index." << endl;
    }

    return outputVector;
}

ReadTrajectory::ReadTrajectory(int argc, char *argv[])
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

ReadTrajectory::~ReadTrajectory()//No destructor?
{
}

void ReadTrajectory::quit()//does not return?
{
}

int ReadTrajectory::compute(const char *port)
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
ReadTrajectory::readFile()
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
    vector<Point> point_list;
    
    // read one line after another
    while (fgets(line, LINE_SIZE, fp) != NULL)
    {
        // read the keyword
        numScanned = sscanf(line, "%d %d %99s %d",
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
                cerr << "ReadTrajectory::readFile:: sscanf1 failed" << endl;
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
            timeStamp_list.push_back(relativeTimeZero - TimeStamp);
            // Append numCoord to corner list
            corner_list.push_back(numCoords - 1);
            //vector list of points with corresponding time stamp
            point_list.push_back(Point(xValue, yValue, zCoord, (relativeTimeZero - TimeStamp),((corner_list.size())-1),((line_list.size())-1)));

        } else
        {
            cerr << "ReadTrajectory::readFile:: sscanf failed" << endl;
        }
    }

    float radius = 0.02;
    float startTime = 0.0;
    float endTime = 60.0;

    vector<pair<size_t, size_t>> intersectingIndices = findPointsWithinRadius(point_list, radius, startTime, endTime);

    vector<float> x_coordinates_temp;
    vector<float> y_coordinates_temp;
    vector<float> z_coordinates_temp;
    vector<int> corner_list_temp, temp1, temp2;
    vector<int> line_list_temp;
  
    for (const auto& indices : intersectingIndices) 
    {
        size_t i = indices.first;
        size_t j = indices.second;

        int start = point_list[i].getLine();
        int end = start+1;
        line_list_temp.push_back(line_list[start]);
        temp1 = getElementsInRange(corner_list, start, end);
        corner_list_temp.insert(corner_list_temp.end(), temp1.begin(), temp1.end());

        int start2 = point_list[i].getLine();
        int end2 = start2+1;
        line_list_temp.push_back(line_list[start2]);
        temp2 = getElementsInRange(corner_list, start2, end);
        corner_list_temp.insert(corner_list_temp.end(), temp2.begin(), temp2.end());
    }

    for (const auto& indices : corner_list_temp) 
    {
      x_coordinates_temp.push_back(x_coordinates[indices]);
      y_coordinates_temp.push_back(y_coordinates[indices]);
      z_coordinates_temp.push_back(z_coordinates[indices]);
    }

    
    sendInfo("found %d coordinates, %d corners, %d lines", numCoords, numCorners, numLines);

    // get the COVISE output object name from the controller
    lineObjectName = linePort->getObjName();

    // create the COVISE output object
    /*Creating pointer to the first element in the vector
    Do not need to dealocate memory since no new array is created and memory
    allocated for the vector will be automatically deallocated by its destructor*/
    float* cx = x_coordinates_temp.data();
    float* cy = y_coordinates_temp.data();
    float* cz = z_coordinates_temp.data();
    int*   ci = corner_list_temp.data();
    int*   li = line_list_temp.data();
    
    lineObject = new coDoLines(lineObjectName, (x_coordinates_temp.size()), cx, cy, cz, (corner_list_temp.size()), ci, (line_list_temp.size()), li);
    linePort->setCurrentObject(lineObject);
   
}

bool ReadTrajectory::openFile()
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




MODULE_MAIN(IO, ReadTrajectory)
