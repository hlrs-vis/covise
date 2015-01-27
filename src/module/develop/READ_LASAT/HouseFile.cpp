/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   HouseFile
//
// Description:
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: HouseFile.cpp,v 1.3 2002/12/17 13:36:05 ralf Exp $
//
#include "HouseFile.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <util/Triangulator.h>
#ifdef __linux
#include <unistd.h>
#endif
//#include "coDoPolygons.h"
#define M_PI 3.14159265
#include <fstream>
int mystrlen(const char *str)
{
    const char *a = str;
    int len = 0;
    if (NULL == a)
    {
        return 0;
    }
    while (*a++ != '\0')
    {
        len++;
    }
    return len;
}

HouseFile::HouseFile(const char *fileName, float zbase, const char *o_name)
{
    int nameLen = mystrlen(o_name);
    obj_name = new char[1 + nameLen];
    strcpy(obj_name, o_name);

    zbase_ = zbase;

    num_nodes = 0;
    num_polygons = 0;
    num_components = 0;
    ifstream input(fileName); // input file stream
    if (input.fail())
    {
        return;
    }

    char buffer[MAXLINE];
    char line[MAXLINE];
    float x, y, length, width, height, angle;
    char tit[MAXLINE];
    char type[20], type_def[20];
    char height_str[20];

    while (input.getline(buffer, MAXLINE))
    {
        if (buffer[0] != '*' && buffer[0] != '-' && buffer[0] != '!')
        {
            strcpy(type, "begin");
            sscanf(buffer, "%s %s %s", type_def, height_str, type);

            if (strstr(type_def, "Btype") != NULL)
            {
                if (strstr(type, "BOX") != NULL)
                {
                    input.getline(line, MAXLINE); //skip
                    input.getline(line, MAXLINE); //skip
                    input.getline(line, MAXLINE); //skip
                    input.getline(line, MAXLINE); //skip

                    input.getline(line, MAXLINE);

                    while (line[0] != '*' && line[0] != '-' && line[0] != '!')
                    {
                        strcpy(tit, strtok(line, " "));
                        sscanf(strtok(NULL, " "), "%f", &x);
                        sscanf(strtok(NULL, " "), "%f", &y);
                        sscanf(strtok(NULL, " "), "%f", &length);
                        sscanf(strtok(NULL, " "), "%f", &width);
                        sscanf(strtok(NULL, " "), "%f", &height);
                        sscanf(strtok(NULL, " "), "%f", &angle);
#ifdef DEBUG
                        cerr << tit << " " << x << " " << y << " " << length << " " << angle << endl;
#endif
                        addBox(x, y, zbase, length, width, height, angle);

                        if (!input.getline(line, MAXLINE))
                        {
                            break;
                        }
                    }
                }
                else if (strstr(type, "POLY") != NULL)
                {
                    input.getline(line, MAXLINE);
                    strtok(line, " ");
                    strtok(NULL, " ");
                    sscanf(strtok(NULL, " "), "%f", &height);

                    float x[200], y[200];
                    int num_vert = 0; // number of vertexes in polygon

                    input.getline(line, MAXLINE); //skip
                    input.getline(line, MAXLINE); //skip
                    input.getline(line, MAXLINE); //skip

                    input.getline(line, MAXLINE);

                    while (line[0] != '*' && line[0] != '-' && line[0] != '!' && line[0] != '\r')
                    {
                        strcpy(tit, strtok(line, " "));
                        strtok(NULL, " ");
                        sscanf(strtok(NULL, " "), "%f", &x[num_vert]);
                        sscanf(strtok(NULL, " "), "%f", &y[num_vert]);
#ifdef DEBUG
                        cerr << "POLY - Point :"
                             << " " << x[num_vert] << " " << y[num_vert] << endl;
#endif
                        num_vert++;

                        if (!input.getline(line, MAXLINE))
                        {
                            break;
                        }
                    }

                    addPrism(num_vert - 1, x, y, height);
                }
            }
        }
    }
    input.close();
}

void
HouseFile::addBox(int ox,
                  int oy,
                  int oz,
                  int length,
                  int width,
                  int height,
                  int angle)
{
    int vertexList[24] = { 0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4 };
    int polygonList[6] = { 0, 4, 8, 12, 16, 20 };
    int i;

    // update polygon list
    for (i = 0; i < 6; i++)
    {
        pl[num_polygons++] = polygonList[i] + num_components;
    }

    // update corner list
    for (i = 0; i < 24; i++)
    {
        cl[num_components++] = vertexList[i] + num_nodes;
    }

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   . height(z)
    //  .       .   .
    //  .   1   .   2
    //  .       . . width(y)
    //  0.......3
    //     length(x)

    xCoords[num_nodes + 0] = ox;
    yCoords[num_nodes + 0] = oy;
    zCoords[num_nodes + 0] = oz;

    xCoords[num_nodes + 1] = ox;
    yCoords[num_nodes + 1] = oy + width;
    zCoords[num_nodes + 1] = oz;

    xCoords[num_nodes + 2] = ox + length;
    yCoords[num_nodes + 2] = oy + width;
    zCoords[num_nodes + 2] = oz;

    xCoords[num_nodes + 3] = ox + length;
    yCoords[num_nodes + 3] = oy;
    zCoords[num_nodes + 3] = oz;

    xCoords[num_nodes + 4] = ox;
    yCoords[num_nodes + 4] = oy;
    zCoords[num_nodes + 4] = oz + height;

    xCoords[num_nodes + 5] = ox;
    yCoords[num_nodes + 5] = oy + width;
    zCoords[num_nodes + 5] = oz + height;

    xCoords[num_nodes + 6] = ox + length;
    yCoords[num_nodes + 6] = oy + width;
    zCoords[num_nodes + 6] = oz + height;

    xCoords[num_nodes + 7] = ox + length;
    yCoords[num_nodes + 7] = oy;
    zCoords[num_nodes + 7] = oz + height;

    // rotate
    if (angle != 0.0)
    {
        double rot_sin = sin((angle * M_PI) / 180.0);
        double rot_cos = cos((angle * M_PI) / 180.0);
        float x, y;

        for (i = num_nodes; i < num_nodes + 8; i++)
        {
            x = xCoords[i] - ox;
            y = yCoords[i] - oy;

            xCoords[i] = ox + x * rot_cos - y * rot_sin;
            yCoords[i] = oy + x * rot_sin + y * rot_cos;
        }
    }

    num_nodes += 8;
}

/* lines_intersect:  AUTHOR: Mukesh Prasad
 *
 *   This function computes whether two line segments,
 *   respectively joining the input points (x1,y1) -- (x2,y2)
 *   and the input points (x3,y3) -- (x4,y4) intersect.
 *   If the lines intersect, the output variables x, y are
 *   set to coordinates of the point of intersection.
 *
 *   All values are in integers.  The returned value is rounded
 *   to the nearest integer point.
 *
 *   If non-integral grid points are relevant, the function
 *   can easily be transformed by substituting floating point
 *   calculations instead of integer calculations.
 *
 *   Entry
 *        x1, y1,  x2, y2   Coordinates of endpoints of one segment.
 *        x3, y3,  x4, y4   Coordinates of endpoints of other segment.
 *
 *   Exit
 *        x, y              Coordinates of intersection point.
 *
 *   The value returned by the function is one of:
 *
 *        DONT_INTERSECT    0
 *        DO_INTERSECT      1
 *        COLLINEAR         2
 *
 * Error conditions:
 *
 *     Depending upon the possible ranges, and particularly on 16-bit
 *     computers, care should be taken to protect from overflow.
 *
 *     In the following code, 'long' values have been used for this
 *     purpose, instead of 'int'.
 *
 */

const int DONT_INTERSECT = 0;
const int DO_INTERSECT = 1;
const int COLLINEAR = 2;

bool SAME_SIGNS(double a, double b)
{
    return (((a < 0.0) && (b < 0.0)) || ((a >= 0.0) && (b >= 0.0)));
}

int lines_intersect(double x1, double y1, /* First line segment */
                    double x2, double y2,

                    double x3, double y3, /* Second line segment */
                    double x4, double y4,

                    double &x,
                    double &y
                    /* Output value:
 * point of intersection */
                    )
{
    double a1, a2, b1, b2, c1, c2; /* Coefficients of line eqns. */
    double r1, r2, r3, r4; /* 'Sign' values */
    double denom, offset, num; /* Intermediate values */

    /* Compute a1, b1, c1, where line joining points 1 and 2
    * is "a1 x  +  b1 y  +  c1  =  0".
    */

    a1 = y2 - y1;
    b1 = x1 - x2;
    c1 = x2 * y1 - x1 * y2;

    /* Compute r3 and r4.
    */

    r3 = a1 * x3 + b1 * y3 + c1;
    r4 = a1 * x4 + b1 * y4 + c1;

    /* Check signs of r3 and r4.  If both point 3 and point 4 lie on
    * same side of line 1, the line segments do not intersect.
    */

    if (r3 != 0 && r4 != 0 && SAME_SIGNS(r3, r4))
        return (DONT_INTERSECT);

    /* Compute a2, b2, c2 */

    a2 = y4 - y3;
    b2 = x3 - x4;
    c2 = x4 * y3 - x3 * y4;

    /* Compute r1 and r2 */

    r1 = a2 * x1 + b2 * y1 + c2;
    r2 = a2 * x2 + b2 * y2 + c2;

    /* Check signs of r1 and r2.  If both point 1 and point 2 lie
    * on same side of second line segment, the line segments do
    * not intersect.
    */

    if (r1 != 0 && r2 != 0 && SAME_SIGNS(r1, r2))
        return (DONT_INTERSECT);

    /* Line segments intersect: compute intersection point.
    */

    denom = a1 * b2 - a2 * b1;
    if (denom == 0)
        return (COLLINEAR);
    offset = denom < 0 ? -denom / 2 : denom / 2;

    /* The denom/2 is to get rounding instead of truncating.  It
    * is added or subtracted to the numerator, depending upon the
    * sign of the numerator.
    */

    num = b1 * c2 - b2 * c1;
    x = (num < 0 ? num - offset : num + offset) / denom;

    num = a2 * c1 - a1 * c2;
    y = (num < 0 ? num - offset : num + offset) / denom;

    return (DO_INTERSECT);
} /* lines_intersect */

void
HouseFile::addPrism(int numVert,
                    float *x,
                    float *y,
                    float height)
{

    int(*triangles)[3] = new int[1 + 2 * numVert][3];

    int numTria = Triangulator::getTriangles(numVert, x, y, triangles);
    int i;
    float xx[3];
    float yy[3];
#ifdef DEBUG
    cerr << "numVert is:" << numVert << "  numTria is:" << numTria << endl;
#endif
    for (i = 0; i < numTria; i++)
    {
        xx[0] = x[triangles[i][0]];
        yy[0] = y[triangles[i][0]];
        xx[1] = x[triangles[i][1]];
        yy[1] = y[triangles[i][1]];
        xx[2] = x[triangles[i][2]];
        yy[2] = y[triangles[i][2]];
        addPrism_(3, xx, yy, height);
    }
    delete[] triangles;
}

void
HouseFile::addPrism_(int numVert,
                     float *x,
                     float *y,
                     float height)
{
    int i, j;
    int old_num_nodes = num_nodes;

    //top and bottom of prism
    for (j = 0; j < 2; j++)
    {
        pl[num_polygons++] = num_components;

        for (i = 0; i < numVert; i++)
        {
            cl[num_components++] = num_nodes;

            xCoords[num_nodes] = x[i];
            yCoords[num_nodes] = y[i];
            zCoords[num_nodes++] = (j == 0) ? zbase_ : height;
        }
    }

    // the rest
    for (i = 0; i < numVert - 1; i++)
    {
        pl[num_polygons++] = num_components;

        cl[num_components++] = old_num_nodes + i;
        cl[num_components++] = old_num_nodes + i + 1;
        cl[num_components++] = old_num_nodes + i + numVert + 1;
        cl[num_components++] = old_num_nodes + i + numVert;
    }
    pl[num_polygons++] = num_components;
    cl[num_components++] = old_num_nodes + numVert - 1;
    cl[num_components++] = old_num_nodes;
    cl[num_components++] = old_num_nodes + numVert;
    cl[num_components++] = old_num_nodes + 2 * numVert - 1;
}

coDistributedObject *
HouseFile::getPolygon()
{
    coDoPolygons *poly = new coDoPolygons(obj_name, num_nodes, xCoords, yCoords, zCoords, num_components, cl,
                                          num_polygons, pl);

    if (poly && poly->objectOk())
    {
        poly->addAttribute("vertexOrder", "2");
        return poly;
    }

    return NULL;
}

//
// History:
//
// $Log: HouseFile.cpp,v $
// Revision 1.3  2002/12/17 13:36:05  ralf
// adapted for windows
//
// Revision 1.2  2002/12/16 14:16:02  cs_te
// -
//
// Revision 1.1  2002/12/12 11:58:58  cs_te
// initial version
//
//
