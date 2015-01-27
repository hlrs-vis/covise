/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <fstream>
#include "Functions.h"
#include <math.h>

using namespace std;

void exportHmo(Plane myPlane, const char *planeFile)
{
    ofstream out_file(planeFile);
    list<Node>::iterator nodeList_Iter;

    out_file << "#COMP_DATA#\n";
    out_file << "1\n";
    out_file << "1             MySurface\n";
    out_file << "#END#\n";
    out_file << "#NODL_DATA#\n";
    list<Node>::iterator nodeList_endIter = myPlane._nodeList.end();
    nodeList_endIter--;
    out_file << nodeList_endIter->number << "\n";
    nodeList_Iter = myPlane._nodeList.begin();
    for (int i = 0; i <= (int)myPlane._nodeList.size() - 1; i++)
    {

        if ((*nodeList_Iter).number != -1)
        {
            out_file << "   ";
            out_file << (*nodeList_Iter).number << " ";
            out_file << (*nodeList_Iter).x << " ";
            out_file << (*nodeList_Iter).y << " ";
            out_file << (*nodeList_Iter).z << '\n';
            nodeList_Iter++;
        }
        else
            nodeList_Iter++;
    }
    out_file << "#END#\n";
    out_file << "#ELEM_DATA#\n";
    out_file << "108,1:    " << myPlane._elementList.size() << "\n";

    list<Element>::iterator elementList_Iter;
    elementList_Iter = myPlane._elementList.begin();
    for (int i = 0; i <= (int)myPlane._elementList.size() - 1; i++)
    {
        out_file << "   ";
        out_file << (*elementList_Iter).number << " ";
        out_file << 1 << " ";
        //The order is from the bottomLeft node to the topLeft node
        out_file << (*elementList_Iter).numberofNode[0] << " ";
        out_file << (*elementList_Iter).numberofNode[1] << " ";
        out_file << (*elementList_Iter).numberofNode[2] << " ";
        out_file << (*elementList_Iter).numberofNode[4] << " ";
        out_file << (*elementList_Iter).numberofNode[7] << " ";
        out_file << (*elementList_Iter).numberofNode[6] << " ";
        out_file << (*elementList_Iter).numberofNode[5] << " ";
        out_file << (*elementList_Iter).numberofNode[3] << "\n";
        elementList_Iter++;
    }
    out_file << "#END#\n";
    out_file << "#BDRY_DATA#\n";
    out_file << "0\n";
    out_file << "#END#\n";
    out_file.flush();
    out_file.close();
}

void exportHmascii(Plane myPlane)
{
    char fileName[] = "D:\\User_Data\\itezhji\\rechnungen\\results\\plane_.hmascii";
    std::ofstream out_file(fileName);
    list<Node>::iterator nodeList_Iter;
    nodeList_Iter = myPlane._nodeList.begin();
    out_file << "*filetype(ASCII)\n";
    out_file << "*version(4.0)\n";
    out_file << "BEGIN DATA\n";
    out_file << "\n";
    out_file << "BEGIN NODES\n";
    out_file << "#NODL_DATA#\n";

    for (int i = 0; i <= (int)myPlane._nodeList.size() - 1; i++)
    {

        if ((*nodeList_Iter).number != -1)
        {
            out_file << "*node(";
            out_file << (*nodeList_Iter).number << ",";
            out_file << (*nodeList_Iter).x << ",";
            out_file << (*nodeList_Iter).y << ",";
            out_file << (*nodeList_Iter).z << ",0,0,0)\n";
            nodeList_Iter++;
        }
        else
            nodeList_Iter++;
    }
    out_file << "#END NODES\n";
    out_file << "\n";
    out_file << "BEGIN COMPONENTS\n";
    out_file << "*component(1,"
                "My_Surface"
                ",0,4)\n";

    list<Element>::iterator elementList_Iter;
    elementList_Iter = myPlane._elementList.begin();
    for (int i = 0; i <= (int)myPlane._elementList.size() - 1; i++)
    {
        out_file << "*quad8(";
        out_file << (*elementList_Iter).number << ",";
        out_file << "1,";
        //The conner nodes from bottom left to the upper left
        out_file << (*elementList_Iter).numberofNode[0] << ",";
        out_file << (*elementList_Iter).numberofNode[2] << ",";
        out_file << (*elementList_Iter).numberofNode[7] << ",";
        out_file << (*elementList_Iter).numberofNode[5] << ",";
        //the middle nodes form the bottom middle to the top middle
        out_file << (*elementList_Iter).numberofNode[1] << ",";
        out_file << (*elementList_Iter).numberofNode[4] << ",";
        out_file << (*elementList_Iter).numberofNode[6] << ",";
        out_file << (*elementList_Iter).numberofNode[3] << ")\n";
        elementList_Iter++;
    }
    out_file << "END COMPONENTS\n";
    out_file << "\n";
    out_file << "END DATA\n";
    out_file.close();
}

//correct the plane.hmo, place a blankspace between 108,1: and number of the elements"
//firstFile is the components file, the second plane file

void filesBinding(const char *firstFile, const char *secondFile, const char *thirdFile, const char *targetFile)
{
    ifstream in_FirstFile(firstFile);
    ifstream in_SecondFile(secondFile);
    ifstream in_ThirdFile(thirdFile);
    ofstream out_TargetFile(targetFile);

    char firstBuffer[256];
    char secondBuffer[256];
    char thirdBuffer[256];

    int numberOfNodes1 = 0, numberOfNodes2 = 0, numberOfNodes3 = 0, numberOfElements1 = 0, numberOfElements2 = 0, numberOfElements3 = 0;
    int counter1 = 0, counter2 = 0, counter3 = 0;

    //extract the number of nodes and elements from the first input files
    while (true)
    {
        counter1++;

        in_FirstFile.getline(firstBuffer, 256);

        if (counter1 == 8)
        {
            numberOfNodes1 = atoi(firstBuffer);
        }

        if ((counter1 != 0) & (counter1 == numberOfNodes1 + 10))
        {
            in_FirstFile.get(firstBuffer, 7);
            in_FirstFile.getline(firstBuffer, 256);
            numberOfElements1 = atoi(firstBuffer);
            break;
        }
    }
    //extract the number of nodes and elements from the second input files

    while (true)
    {
        counter2++;
        in_SecondFile.getline(secondBuffer, 256);
        if (counter2 == 6)
        {
            numberOfNodes2 = atoi(secondBuffer);
        }
        if ((counter2 != 0) && (counter2 == numberOfNodes2 + 8))
        {
            in_SecondFile.get(secondBuffer, 7);
            in_SecondFile.getline(secondBuffer, 256);
            numberOfElements2 = atoi(secondBuffer);
            break;
        }
    }
    //extract the number of nodes and elements from the third input files

    while (true)
    {
        counter3++;
        in_ThirdFile.getline(thirdBuffer, 256);
        if (counter3 == 6)
        {
            numberOfNodes3 = atoi(thirdBuffer);
        }
        if ((counter3 != 0) && (counter3 == numberOfNodes3 + 8))
        {
            in_ThirdFile.get(thirdBuffer, 7);
            in_ThirdFile.getline(thirdBuffer, 256);
            numberOfElements3 = atoi(thirdBuffer);
            break;
        }
    }

    in_FirstFile.seekg(0, ios::beg);
    in_SecondFile.seekg(0, ios::beg);
    in_ThirdFile.seekg(0, ios::beg);

    //combine the three files into targer files

    // write the nodes data in the first file into the target file
    for (int i = 1; i <= numberOfNodes1 + 10; i++)
    {
        if ((i != 8) & (i <= numberOfNodes1 + 8))
        {
            if (i == 2)
            {
                in_FirstFile.getline(firstBuffer, 256);
                out_TargetFile << "       4\n";
                out_TargetFile << "       1                           Elektrode_1\n";
            }
            else
            {
                in_FirstFile.getline(firstBuffer, 256);
                out_TargetFile << firstBuffer << "\n";
            }
        }
        else if (i > numberOfNodes1 + 8)
            in_FirstFile.getline(firstBuffer, 256);
        else
        {
            //snprintf(firstBuffer, sizeof(firstBuffer), "%10d", numberOfNodes1+numberOfNodes2);
            out_TargetFile << (numberOfNodes1 + numberOfNodes2 + numberOfNodes3) << "\n";
            in_FirstFile.getline(firstBuffer, 256);
        }
    }

    // write the nodes data in the second file into the target file
    for (int i = 1; i <= numberOfNodes2 + 8; i++)
    {
        if (i > 6)
        {
            if (i <= numberOfNodes2 + 6)
            {
                in_SecondFile.getline(secondBuffer, 256);

                out_TargetFile << secondBuffer << "\n";
            }
            else
                in_SecondFile.getline(secondBuffer, 256);
        }
        else
        {
            in_SecondFile.getline(secondBuffer, 256);
        }
    }
    // write the nodes data in the third file into the target file
    for (int i = 1; i <= numberOfNodes3 + 8; i++)
    {
        if (i > 6)
        {
            in_ThirdFile.getline(thirdBuffer, 256);

            out_TargetFile << thirdBuffer << "\n";
        }
        else
        {
            in_ThirdFile.getline(thirdBuffer, 256);
        }
    }
    //out_TargetFile.Flush();

    //skip the useless data
    in_ThirdFile.getline(thirdBuffer, 256);
    in_SecondFile.getline(secondBuffer, 256);
    in_FirstFile.getline(firstBuffer, 256);
    //char helpString[256];
    out_TargetFile << "108,1:   ";
    //snprintf(helpString, sizeof(helpString), "%10d", numberOfNodes1+numberOfNodes2);
    out_TargetFile << (numberOfElements1 + numberOfElements2 + numberOfElements3) << "\n";
    //write the elements data in the first file into the target file
    for (int i = 1; i <= numberOfElements1; i++)
    {
        in_FirstFile.getline(firstBuffer, 256);
        out_TargetFile << firstBuffer << "\n";
    }
    //write the elements data in the second file into the target file

    for (int i = 1; i <= numberOfElements2; i++)
    {
        in_SecondFile.getline(secondBuffer, 256);
        out_TargetFile << secondBuffer << "\n";
    }
    //write the elements data in the third file into the target file

    for (int i = 1; i <= numberOfElements3; i++)
    {
        in_ThirdFile.getline(thirdBuffer, 256);
        out_TargetFile << thirdBuffer << "\n";
    }
    //write the rest part of the file1 into the target file

    while (in_FirstFile.eof())
    {
        in_FirstFile.getline(firstBuffer, 256);
        out_TargetFile << firstBuffer << "\n";
    }
    out_TargetFile.flush();
    in_FirstFile.close();
    in_SecondFile.close();
    in_ThirdFile.close();
    out_TargetFile.close();
}

bool FileExists(const char *filename)
{

    ifstream file;
    file.open(filename);

    // Check if the file exists
    if (file.is_open() == true)
    {
        file.close();
        return true;
    }

    return false;
}

//rotate a point around a center point at XY,YZ or ZX

void rotatePoint(Node *origNode, Node centerNode, float degree, int axis)
{
    float radius = 0;
    double pi = 3.1415926535;
    double radian = degree * pi / 180;

    float xDist = 0, yDist = 0, zDist = 0;
    switch (axis)
    {

    // 1 indicts at XY plane
    case 1:

        xDist = origNode->x - centerNode.x;
        yDist = origNode->y - centerNode.y;
        if (yDist >= 0)
            radian += atan2(yDist, xDist);
        if (yDist < 0)
            radian += atan2(yDist, xDist) + 2 * pi;
        radius = sqrt(pow((centerNode.x - origNode->x), 2) + pow((centerNode.y - origNode->y), 2));
        origNode->x = radius * cos(radian) + centerNode.x;
        origNode->y = radius * sin(radian) + centerNode.y;
        break;

    // 2 indicts at YZ plane
    case 2:

        yDist = origNode->y - centerNode.y;
        zDist = origNode->z - centerNode.z;
        if (zDist >= 0)
            radian += atan2(zDist, yDist);
        if (zDist < 0)
            radian += atan2(zDist, yDist) + 2 * pi;
        radius = sqrt(pow((centerNode.y - origNode->y), 2) + pow((centerNode.z - origNode->z), 2));
        origNode->y = radius * cos(radian) + centerNode.y;
        origNode->z = radius * sin(radian) + centerNode.z;
        break;

    // 3 indicts at ZX plane
    case 3:

        zDist = origNode->z - centerNode.z;
        xDist = origNode->x - centerNode.x;
        if (xDist >= 0)
            radian += atan2(xDist, zDist);
        if (xDist < 0)
            radian += atan2(xDist, zDist) + 2 * pi;
        radius = sqrt(pow((centerNode.z - origNode->z), 2) + pow((centerNode.x - origNode->x), 2));
        origNode->z = radius * cos(radian) + centerNode.z;
        origNode->x = radius * sin(radian) + centerNode.x;
        break;
    }
}
void transformBlock(const char *blockFile, const char *tempFile, float *moveVec3, float *scaleVec3)
{
    ifstream In_BlockFile(blockFile);
    ofstream Out_BlockFile(tempFile);
    float predMoveVec3[3] = { 2.875, 0, 8 };
    float predScaleVec3[3] = { 7.125, 1, 1 };
    char stringBuf[256];
    int counter = 0;
    int numOfNodes;
    float xCoord, yCoord, zCoord;
    while (!In_BlockFile.eof())
    {
        counter++;
        if (counter < 6)
        {
            In_BlockFile.getline(stringBuf, 256);
            Out_BlockFile << stringBuf << "\n";
        }
        else if (counter == 6)
        {
            In_BlockFile.getline(stringBuf, 256);
            numOfNodes = atoi(stringBuf);
            Out_BlockFile << stringBuf << "\n";
            for (int i = 0; i < numOfNodes; i++)
            {
                //Number of the Node;
                In_BlockFile >> stringBuf;
                Out_BlockFile << "  " << stringBuf << " ";
                // x Coordinate
                In_BlockFile >> stringBuf;
                xCoord = atof(stringBuf);
                xCoord = xCoord * predScaleVec3[0] * scaleVec3[0] + predMoveVec3[0] + moveVec3[0];
                Out_BlockFile << xCoord << "         ";
                // yCoorinate
                In_BlockFile >> stringBuf;
                yCoord = atof(stringBuf);
                yCoord = yCoord * predScaleVec3[1] * scaleVec3[1] + predMoveVec3[1] + moveVec3[1];
                Out_BlockFile << yCoord << "         ";
                //zCoordinate
                In_BlockFile >> stringBuf;
                zCoord = atof(stringBuf);
                zCoord = zCoord * predScaleVec3[2] * scaleVec3[2] + predMoveVec3[2] + moveVec3[2];
                Out_BlockFile << zCoord;

                In_BlockFile.getline(stringBuf, 256);
                Out_BlockFile << "\n";
            }
        }
        else
        {
            In_BlockFile.getline(stringBuf, 256);
            Out_BlockFile << stringBuf << "\n";
        }
    }
    In_BlockFile.close();
    Out_BlockFile.close();
}
