/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <map>
#include <stdint.h>

#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
namespace alg = __gnu_parallel;
#else
namespace alg = std;
#endif

using namespace std;
bool intensityOnly;

float min_x, min_y, min_z;
float max_x, max_y, max_z;

enum formatTypes
{
    FORMAT_IRGB,
    FORMAT_RGBI,
    FORMAT_UVRGBI
};

struct Point
{
    float x;
    float y;
    float z;
    uint32_t rgba;
    int l;
};

bool sortfunction(Point i, Point j) { return (i.l < j.l); };

void ReadData(char *filename, std::vector<Point> &vec, formatTypes format)
{

    cout << "Input Data: " << filename << endl;

    ifstream file(filename, ios::in | ios::binary);

    int count = 0;
    if (file.is_open())
    {
        uint32_t size;
        file.read((char *)&size, sizeof(uint32_t));
        cerr << "Total num of sets is " << (size) << endl;
        cerr << "max_size: " << vec.max_size() << "\n";
        cerr << "size: " << vec.size() << "\n";
        for (uint32_t i = 0; i < size; i++)
        {
            unsigned int psize;
            file.read((char *)&psize, sizeof(psize));
            printf("Size of set %d is %d\n", i, psize);
            // read point data
            size_t numP = psize;
            size_t numF = 3 * numP;
            printf("numFloats %zu is %d\n", (size_t)i, (int)numF);
            float *coord = new float[numF];
            uint32_t *icolor = new uint32_t[psize];
            file.read((char *)(coord), (sizeof(float) * 3 * psize));
            //read color data
            file.read((char *)(icolor), (sizeof(uint32_t) * psize));
            Point point;
            for (size_t j = 0; j < psize; j++)
            {
                point.x = coord[j * 3];
                point.y = coord[j * 3 + 1];
                point.z = coord[j * 3 + 2];
                point.rgba = icolor[j];
                point.l = 0;
                vec.push_back(point);

                if (point.x < min_x)
                    min_x = point.x;
                if (point.y < min_y)
                    min_y = point.y;
                if (point.z < min_z)
                    min_z = point.z;

                if (point.x > max_x)
                    max_x = point.x;
                if (point.y > max_y)
                    max_y = point.y;
                if (point.z > max_z)
                    max_z = point.z;
            }
            delete[] coord;
            delete[] icolor;
        }
        file.close();
    }
}

void LabelData(int grid, std::vector<Point> &vec, std::map<int, int> &lookUp)
{

    int xl, yl, zl, xs, ys, zs;

    float xsize, ysize, zsize;
    xs = ys = zs = grid;

    xsize = (max_x - min_x) / grid;
    ysize = (max_y - min_y) / grid;
    zsize = (max_z - min_z) / grid;

    // compute preportional grid sizes
    if (xsize <= ysize && xsize <= zsize)
    {
        xs = grid;
        ys = (int)((max_y - min_y) / xsize);
        ysize = (max_y - min_y) / ys;
        zs = (int)((max_z - min_z) / xsize);
        zsize = (max_z - min_z) / zs;
    }
    else if (ysize <= xsize && ysize <= zsize)
    {
        ys = grid;
        xs = (int)((max_x - min_x) / ysize);
        xsize = (max_x - min_x) / xs;
        zs = (int)((max_z - min_z) / ysize);
        zsize = (max_z - min_z) / zs;
    }
    else
    {
        zs = grid;
        ys = (int)((max_y - min_y) / zsize);
        ysize = (max_y - min_y) / ys;
        xs = (int)((max_x - min_x) / zsize);
        xsize = (max_x - min_x) / zs;
    }

    std::map<int, int>::iterator it;

    printf("Number of points is %d\n", (int)vec.size());

    for (int i = 0; i < vec.size(); i++)
    {

        xl = (int)((vec.at(i).x - min_x) / xsize);
        yl = (int)((vec.at(i).y - min_y) / ysize);
        zl = (int)((vec.at(i).z - min_z) / zsize);

        if (xl == xs)
            xl--;
        if (yl == ys)
            yl--;
        if (zl == zs)
            zl--;

        vec.at(i).l = xl + (yl * xs) + (zl * xs * ys);

        it = lookUp.find(vec.at(i).l);
        if (it != lookUp.end())
        {
            (*it).second++;
        }
        else
        {
            lookUp.insert(std::pair<int, int>(vec.at(i).l, 1));
        }
    }

    cout << "Total Number of sets is " << lookUp.size() << endl;

    // randomize all the data
    //std::random_shuffle(vec.begin(), vec.end());
    alg::random_shuffle(vec.begin(), vec.end());

    // sort data
    //std::sort(vec.begin(), vec.end(), sortfunction);
    alg::sort(vec.begin(), vec.end(), sortfunction);
}

void WriteData(char *filename, std::vector<Point> &vec, std::map<int, int> &lookUp, int maxPointsPerCube)
{
    cout << "Output Data: " << filename << endl;

    ofstream file(filename, ios::out | ios::binary | ios::ate);
    int numPointsToWrite;

    if (file.is_open())
    {
        int number_of_sets = (int)lookUp.size();
        int index = 0;

        // write the number of sets
        file.write((char *)&(number_of_sets), sizeof(int));

        for (int i = 0; i < number_of_sets; i++)
        {
            // get first in set to find set number
            Point first = vec.at(index);

            std::map<int, int>::iterator it;
            it = lookUp.find(first.l);
            if (it != lookUp.end())
            {
                int numPoints = (*it).second;
                //printf("Number of points in set is %d\n", numPoints);

                // restrict number of points written
                if (maxPointsPerCube != -1 && maxPointsPerCube < numPoints)
                {
                    numPointsToWrite = maxPointsPerCube;
                }
                else
                {
                    numPointsToWrite = numPoints;
                }

                // write size
                file.write((char *)&(numPointsToWrite), sizeof(int));

                // write points
                for (int j = index; j < numPointsToWrite + index; j++)
                {
                    file.write((char *)&(vec.at(j).x), sizeof(float) * 3);
                }

                // write colors
                for (int j = index; j < numPointsToWrite + index; j++)
                {
                    file.write((char *)&(vec.at(j).rgba), sizeof(uint32_t));
                }

                // increment offset
                index = index + numPoints;
            }
        }
    }
    file.close();

    cout << "Data Written!" << endl;
}

int main(int argc, char **argv)
{

    // TODO these values should be command line arguments
    int maxPointsPerCube = 1550000; // note set to -1 if no max points per cube is specified
    int divisionSize = 25;
    formatTypes format = FORMAT_IRGB;
    std::vector<Point> vec;
    std::map<int, int> lookUp;

    min_x = min_y = min_z = FLT_MAX;
    max_x = max_y = max_z = FLT_MIN;

    intensityOnly=false;
    if (argc < 3) /* argc should be > 3 for correct execution */
    {
        printf("Minimal two params required. read README.txt\n");
    }
    else
    {
        for (int i = 1; i < argc - 1; i++)
        {
            if(argv[i][0] == '-')
            {
               if(argv[i][1] == 'i')
               {
                   intensityOnly=true;
               }
            }
            else
            {
                printf("Reading in %s\n", argv[i]);
                ReadData(argv[i], vec, format);
            }
        }
        printf("Sorting data\n");
        LabelData(divisionSize, vec, lookUp);
        printf("Persisting data\n");
        WriteData(argv[argc - 1], vec, lookUp, maxPointsPerCube);
    }
    return 0;
}
