/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// 19.11.2001 / 1 / file ReadIWS.h

#ifndef _READ_IWS_H
#define _READ_IWS_H

/***************************************************************************\ 
 **                                                           (C)2001 RUS **
 **                                                                       **
 ** Description: Reader for files from IWS                                **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 **                                                                       **
 ** Author: M. Muench                                                     **
 **                                                                       **
 ** History:                                                              **
 ** March 01         v1                                                   **
 ** xxxxxxxx         new covise api                                       **
\***************************************************************************/

/***********************************\ 
 *                                 *
 *  place the #include files here  *
 *                                 *
\***********************************/

#include <api/coModule.h>
using namespace covise;

#include <util/coviseCompat.h>

/*********************************************\ 
 *                                           *
 *  place for all the "elementary" typedefs  *
 *                                           *
\*********************************************/

/****************************\ 
 *                          *
 *  place your macros here  *
 *                          *
\****************************/

//lenght of a line
const int LINE_SIZE = 8192;

//portion for resizing data
const int CHUNK_SIZE = 4096;

const int FIRST_WORD = 1;

const int KEY_SIZE = 100;

class ReadIWS : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    FILE *openFile(const char *filename);
    void readFile(float scale_factor);
    void closeFile(const char *filename);

    void scanHeader(char *line, int *nVert, int *nEdge, int *nCorner, int *nFace, int *nElem);

    //tests, if coordinates, edges, faces or elements.
    char *testInput(char *line);

    //allocators
    void allocCoord(char *line, char *key, int nVert, float **cx, float **cy, float **cz);
    void allocLines(char *line, char *key, int nCorner, int nEdge, int **cl, int **ll);
    void allocPoly(char *line, char *key, int nFace, int **pl, int **id);

    //polygons
    int setPolyId(char *line, char *key);
    int setPolygons(char *line, char *key, int *old_size, int *new_size, int polygon_index, int **polygon_list, int edge_index, int **edge_list);
    void line2coord(int *corner_list, int edge_index, int *edge_list, int polygon_index, int *polygon_list, int **polygon_corners);

    //test if we've got a string representation of a whole number
    //WARNING: IT REALLY CAN BE ANY KIND OF A WHOLE ARBITRARY PRECISION NUMBER !!
    bool isWholeNumber(char *line);

    //test if we've got a string representation of a number
    //WARNING: IT REALLY CAN BE ANY KIND OF A VALID ARBITRARY PRECISION NUMBER !!
    bool isNumber(char *line);
    //true => line shall be skipped as it is a comment, ...
    inline bool skipLine(const char *line)
    {
        for (int i = 0; i < LINE_SIZE; i++)
        {
            if (isspace(*(line + i)))
            {
                continue;
            }
            else
            {
                ;
            }

            if (((*(line + i)) == '#') || ((*(line + i)) == '\0'))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    };

    //  member data
    const char *iwsFile; // openug file name
    FILE *iwsFp;

    //enum { NUMPORTS=6 };

    //ports

    coOutputPort *gridPort;
    coOutputPort *polyPort;
    coOutputPort *linePort;
    coOutputPort *gridDataPort;
    coOutputPort *polyDataPort;
    coOutputPort *lineDataPort;

    //parameters

    //coChoiceParam *selectData[NUMPORTS];
    coFileBrowserParam *iwsFileParam;
    coFloatParam *scaleWorld;

public:
    ReadIWS();
    virtual ~ReadIWS();
};
#endif
