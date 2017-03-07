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
#include <osg/Matrix>
#include <osg/Vec3>

#include <stdint.h>

using namespace std;

float min_x, min_y, min_z;
float max_x, max_y, max_z;
bool intensityOnly;

enum formatTypes
{
    FORMAT_IRGB,
    FORMAT_RGBI,
    FORMAT_If,
    FORMAT_RGB,
    FORMAT_UVRGBI
};

struct Point
{
    float x;
    float y;
    float z;
    uint32_t rgba;
};

void ReadData(char *filename, std::vector<Point> &vec, formatTypes format)
{

    FILE *inputFile;
    cout << "Input Data: " << filename << endl;

    inputFile = fopen(filename, "r");

    if (inputFile == NULL)
    {
        cout << "Error opening file:" << filename;
        return;
    }

    int in, r, g, b, u, v;
    float rf, gf, bf;
    Point point;

    char buf[1000];
    int toRead = 1000;
    if (format == FORMAT_UVRGBI)
        toRead = 8;
    else if (format == FORMAT_IRGB)
        toRead = 7;
    else if (format == FORMAT_RGBI)
        toRead = 6;
    else if (format == FORMAT_If)
        toRead = 4;
    else if (format == FORMAT_RGB)
        toRead = 6;

    while (fgets(buf, 1000, inputFile) != NULL)
    {
        if(buf[0]=='/')
        {
            if(strcmp(buf,"//X,Y,Z,Scalar field,R,G,B")==0)
            {
            fgets(buf, 1000, inputFile); // num points
            fgets(buf, 1000, inputFile);
            }
        }
        int numValues;
        if (format == FORMAT_UVRGBI)
            numValues = sscanf(buf, "%d %d %f %f %f %d %d %d", &u, &v, &(point.x), &(point.y), &(point.z), &r, &g, &b);
        else if (format == FORMAT_IRGB)
        {
            char *c = buf;
            char *nc = c;
            numValues = 0;
#ifdef WIN32
            point.x = strtod(c, &nc);
#else
            point.x = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
#ifdef WIN32
            point.y = strtod(c, &nc);
#else
            point.y = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
#ifdef WIN32
            point.z = strtod(c, &nc);
#else
            point.z = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
#ifdef WIN32
            in = strtod(c, &nc);
#else
            in = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
            r = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
            g = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
            b = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            if(*c == ',')
                c++;
            if(*c == ';')
                c++;
            //numValues =sscanf (buf, "%f %f %f %f %d %d %d", &(point.x),&(point.y),&(point.z), &in, &r, &g, &b);
        }
        else if (format == FORMAT_RGBI)
            numValues = sscanf(buf, "%f %f %f %d %d %d", &(point.x), &(point.y), &(point.z), &r, &g, &b);
        else if (format == FORMAT_If)
        {
            numValues = sscanf(buf, "%f %f %f %f", &(point.x), &(point.y), &(point.z), &rf);
            g = (int)((rf/2000.0) * 255); 
            b = (int)((rf/2000.0) * 255); 
            r = (int)((rf/2000.0) * 255); 
        }
        else if (format == FORMAT_RGB)
        {

            numValues = sscanf(buf, "%f %f %f %f %f %f", &(point.x), &(point.y), &(point.z), &rf, &gf, &bf);
            r = (int)(rf * 255);
            g = (int)(gf * 255);
            b = (int)(bf * 255);
        }
        if (numValues == toRead && !(point.x == 0.0 && point.y == 0.0 && point.z == 0.0))
        {
            point.rgba = r | g << 8 | b << 16;
            vec.push_back(point);
        }
    }

    fclose(inputFile);
}

void ReadPTX(char *filename, std::vector<Point> &vec)
{

    FILE *inputFile;
    cout << "Input Data: " << filename << endl;

    inputFile = fopen(filename, "r");

    if (inputFile == NULL)
    {
        cout << "Error opening file:" << filename;
        return;
    }

    int r, g, b;
    float fa, fb, fc;
    Point point;

    char buf[1000];
    int toRead = 7;
    bool readHeader = true;
    int numRows;
    int numCols;
    unsigned int numLines;
    unsigned int linesRead;
    osg::Matrix m;
    m.makeIdentity();
    while (fgets(buf, 1000, inputFile) != NULL)
    {
        if (readHeader)
        {
            sscanf(buf, "%d", &numRows);
            fgets(buf, 1000, inputFile);
            sscanf(buf, "%d", &numCols);
            fgets(buf, 1000, inputFile);
            fgets(buf, 1000, inputFile);
            fgets(buf, 1000, inputFile);
            fgets(buf, 1000, inputFile); // scanner pos.
            fgets(buf, 1000, inputFile); // 4x4 Matrix
            sscanf(buf, "%f %f %f", &fa, &fb, &fc);
            m(0, 0) = fa;
            m(1, 0) = fb;
            m(2, 0) = fc;
            fgets(buf, 1000, inputFile);
            sscanf(buf, "%f %f %f", &fa, &fb, &fc);
            m(0, 1) = fa;
            m(1, 1) = fb;
            m(2, 1) = fc;
            fgets(buf, 1000, inputFile);
            sscanf(buf, "%f %f %f", &fa, &fb, &fc);
            m(0, 2) = fa;
            m(1, 2) = fb;
            m(2, 2) = fc;
            fgets(buf, 1000, inputFile); // scanner pos.
            sscanf(buf, "%f %f %f", &fa, &fb, &fc);
            m(0, 3) = fa;
            m(1, 3) = fb;
            m(2, 3) = fc;
            readHeader = false;
            linesRead = 0;
            numLines = numRows * numCols;

            printf("Reading in %d lines\n", numLines);
        }
        else
        {
            linesRead++;
            if (linesRead == numLines)
            {
                fprintf(stderr,"Restart %d %d %d\n",linesRead,numRows,numCols);
                readHeader = true;
            }

            int numValues;

            char *c = buf;
            char *nc = c;
            numValues = 0;
#ifdef WIN32
            point.x = strtod(c, &nc);
#else
            point.x = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
#ifdef WIN32
            point.y = strtod(c, &nc);
#else
            point.y = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
#ifdef WIN32
            point.z = strtod(c, &nc);
#else
            point.z = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
#ifdef WIN32
            double in = strtod(c, &nc);
#else
            float in = strtof(c, &nc);
#endif
            if (nc != c)
                numValues++;
            c = nc;
            r = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            g = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            b = strtol(c, &nc, 10);
            if (nc != c)
                numValues++;
            c = nc;
            if (numValues == toRead && !(point.x == 0.0 && point.y == 0.0 && point.z == 0.0))
            {
                osg::Vec3 p(point.x, point.y, point.z);
                p = m * p;
                point.x = p[0];
                point.y = p[1];
                point.z = p[2];
                if(intensityOnly)
                {
                    unsigned char intensity = (unsigned char)(in*255.99);
                    point.rgba = intensity | intensity << 8 | intensity << 16;
                }
                else
                {
                    point.rgba = r | g << 8 | b << 16;
                }
                vec.push_back(point);
            }
        }
    }

    fclose(inputFile);
}

void WriteData(char *filename, std::vector<Point> &vec)
{
    cout << "Output Data: " << filename << endl;

    ofstream file(filename, ios::out | ios::binary | ios::ate);

    if (file.is_open())
    {
        int number_of_sets = 1;
        int index = 0;

        //printf("Vector size is %d\n", vec->size());
        //printf("Number of sets is %d\n", number_of_sets);

        // write the number of sets
        file.write((char *)&(number_of_sets), sizeof(int));

        uint32_t numPoints = vec.size();
        file.write((char *)&(numPoints), sizeof(uint32_t));
        for (int i = 0; i < vec.size(); i++)
        {
            file.write((char *)&(vec.at(i).x), sizeof(float) * 3);
        }
        for (int i = 0; i < vec.size(); i++)
        {
            file.write((char *)&(vec.at(i).rgba), sizeof(uint32_t));
        }
    }
    file.close();

    cout << "Data Written!" << endl;
}

int main(int argc, char **argv)
{

    // TODO these values should be command line arguments
    int maxPointsPerCube = 250000; // note set to -1 if no max points per cube is specified
    int divisionSize = 16;
    formatTypes format = FORMAT_IRGB;
    //format = FORMAT_RGB;
    std::vector<Point> vec;
    vec.reserve(1000000);

    min_x = min_y = min_z = FLT_MAX;
    max_x = max_y = max_z = FLT_MIN;

    if (argc < 3) /* argc should be > 3 for correct execution */
    {
        printf("Minimal two params required. read README.txt\n");
    }
    else
    {
        for (int i = 1; i < argc - 1; i++)
        {
            printf("Reading in %s\n", argv[i]);
            int len = strlen(argv[i]);
            if((len>1) && argv[i][0] == '-')
            {
               if(argv[i][1] == 'i')
               {
                   intensityOnly=true;
               }
            }
            else
            {
            if ((len > 4) && strcmp((argv[i] + len - 4), ".ptx") == 0)
            {
                ReadPTX(argv[i], vec);
            }
            else if ((len > 4) && strcmp((argv[i] + len - 4), ".xyz") == 0)
            {
                format = FORMAT_RGBI;
                ReadData(argv[i], vec, format);
            }
            else if ((len > 4) && strcmp((argv[i] + len - 4), ".pts") == 0)
            {
                format = FORMAT_If;
                ReadData(argv[i], vec, format);
            }
            else
            {
                ReadData(argv[i], vec, format);
            }
            }
        }
        WriteData(argv[argc - 1], vec);
    }
    return 0;
}
