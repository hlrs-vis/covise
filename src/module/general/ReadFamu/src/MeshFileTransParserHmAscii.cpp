/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ResultsFileParserAscii.h
 * a mesh file parser for changing (transient) meshes in HyperMesh ASCII format.
 */

#include "MeshFileTransParserHmAscii.h" // a mesh file parser for changing (transient) meshes in HyperMesh ASCII format.
#include "ReadFamu.h"
#include "Tools.h" // some helpful tools.
#include <do/coDoUnstructuredGrid.h>

MeshFileTransParserHmAscii::MeshFileTransParserHmAscii(OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfPoints(0)
    , _noOfVertices(0)
    , _vertices(0)
    , _noOfElements(0)
    , _fileHandle(NULL)
    , _pushedBack(false)
    , _c(NULL)
{
}

void MeshFileTransParserHmAscii::getMeshData(
    int *noOfElements, int *noOfVertices, int *noOfPoints,
    int *elementsArr[], int *verticesArr[],
    float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
    int *typesArr[])
{
    *noOfElements = _noOfElements;
    *noOfVertices = _noOfVertices;
    *noOfPoints = _noOfPoints;

    *elementsArr = &_elements[0];
    *verticesArr = &_vertices[0];

    *xPointsArr = &_xPoints[0];
    *yPointsArr = &_yPoints[0];
    *zPointsArr = &_zPoints[0];

    *typesArr = &_types[0];
}

void MeshFileTransParserHmAscii::parseMeshFile(
    std::string filename,
    coRestraint &sel,
    bool subdivide,
    const int &noOfTimeStepToSkip,
    const int &noOfTimeStepsToParse)
{
    (void)noOfTimeStepToSkip;
    (void)noOfTimeStepsToParse;
    // compute parameters
    _fileHandle = fopen(filename.c_str(), "r");
    if (!_fileHandle)
    {
        ERROR1("cannot open mesh file ", filename, _outputHandler);
    }

    _pushedBack = false;
    _xPoints.reserve(1000);
    _yPoints.reserve(1000);
    _zPoints.reserve(1000);
    _vertices.reserve(1000);
    _xPoints.clear();
    _yPoints.clear();
    _zPoints.clear();
    _vertices.clear();
    _elements.clear();
    _types.clear();

    _noOfPoints = 0;
    _noOfVertices = 0;
    _noOfElements = 0;
    while (!feof(_fileHandle))
    {
        getLine();
        if (strncmp(_c, "*component(", 11) == 0) // skip unselected components
        {
            int componentID;
            sscanf(_c + 11, "%d", &componentID);
            while (*_c && (*_c != '"'))
                _c++;
            if (*_c)
            {
                _c++;
                const char *componentName = _c;
                while (*_c && (*_c != '"'))
                    _c++;
                if (*_c == '"')
                {
                    *_c = '\0';
                }
                char buf[1000];
                sprintf(buf, "component %d %s\n", componentID, componentName);
                _outputHandler->displayString(buf);
            }
            while (!feof(_fileHandle) && !sel(componentID))
            {
                getLine();
                if (strncmp(_c, "*component(", 11) == 0)
                {
                    sscanf(_c + 11, "%d", &componentID);
                    while (*_c && (*_c != '"'))
                        _c++;
                    if (*_c)
                    {
                        _c++;
                        const char *componentName = _c;
                        while (*_c && (*_c != '"'))
                            _c++;
                        if (*_c == '"')
                        {
                            *_c = '\0';
                        }
                        char buf[1000];
                        sprintf(buf, "component %d %s\n", componentID, componentName);
                        _outputHandler->displayString(buf);
                    }
                }
            }
        }
        if (strncmp(_c, "*quad4(", 7) == 0)
        { // new _line node
            int tetNum;
            int vert[4];
            int iret = sscanf(_c + 7, "%d,1,%d,%d,%d,%d", &tetNum, vert, vert + 1, vert + 2, vert + 3);
            if (iret != 5)
            {
                cerr << "parse error reading quad4 " << iret << endl;
                break;
            }
            // subdiv. not implemented yet
            //if(subdivide)
            //{
            //}
            {
                _elements[_noOfElements] = _noOfVertices;
                _types[_noOfElements] = TYPE_QUAD;
                int i;
                for (i = 0; i < 4; i++)
                {
                    _vertices.push_back(vert[i] - 1);
                    _noOfVertices++;
                }
                _noOfElements++;
            }
        }
        if (strncmp(_c, "*tetra10(", 9) == 0)
        { // new _line node
            int tetNum;
            int vert[10];
            int iret = sscanf(_c + 9, "%d,1,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &tetNum, vert, vert + 1, vert + 2, vert + 3, vert + 4, vert + 5, vert + 6, vert + 7, vert + 8, vert + 9);
            if (iret != 11)
            {
                cerr << "parse error reading tetra10 " << iret << endl;
                break;
            }
            if (subdivide)
            {
                _elements[_noOfElements] = _noOfVertices;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 5;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;

                _vertices.push_back(vert[4] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[7] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[6] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[0] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[4] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[5] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[1] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[5] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[6] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[2] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[7] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[3] - 1);
                _noOfVertices++;

                //innen pyramids

                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[7] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[6] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[5] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[4] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[5] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[6] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[7] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
            }
            else
            {
                _elements[_noOfElements] = _noOfVertices;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                int i;
                for (i = 0; i < 4; i++)
                {
                    _vertices.push_back(vert[i] - 1);
                    _noOfVertices++;
                }
                _noOfElements++;
            }
        }
        if (strncmp(_c, "*hexa20(", 8) == 0)
        { // new _line node
            int tetNum;
            int vert[20];
            int iret = sscanf(_c + 8, "%d,1,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &tetNum, vert, vert + 1, vert + 2, vert + 3, vert + 4, vert + 5, vert + 6, vert + 7, vert + 8, vert + 9, vert + 10, vert + 11, vert + 12, vert + 13, vert + 14, vert + 15, vert + 16, vert + 17, vert + 18, vert + 19);
            if (iret != 21)
            {
                cerr << "parse error reading hexa20 " << iret << endl;
                break;
            }
            if (subdivide)
            {
                _elements[_noOfElements] = _noOfVertices;
                _types[_noOfElements] = TYPE_HEXAEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 8;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 5;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 5;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 5;
                _types[_noOfElements] = TYPE_PYRAMID;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 5;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;
                _elements[_noOfElements] = _elements[_noOfElements - 1] + 4;
                _types[_noOfElements] = TYPE_TETRAHEDER;
                _noOfElements++;

                // hex
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[10] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[11] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[16] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[17] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[18] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[19] - 1);
                _noOfVertices++;

                //pyramids
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[16] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[19] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[11] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[12] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[17] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[16] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[13] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[10] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[18] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[17] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[14] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[10] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[11] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[19] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[18] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[15] - 1);
                _noOfVertices++;

                // tetrahedra

                _vertices.push_back(vert[11] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[12] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[0] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[12] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[16] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[19] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[4] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[8] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[13] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[1] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[13] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[17] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[16] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[5] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[9] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[10] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[14] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[2] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[17] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[14] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[18] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[6] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[10] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[11] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[15] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[3] - 1);
                _noOfVertices++;

                _vertices.push_back(vert[15] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[19] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[18] - 1);
                _noOfVertices++;
                _vertices.push_back(vert[7] - 1);
                _noOfVertices++;
            }
            else
            {
                _elements[_noOfElements] = _noOfVertices;
                _types[_noOfElements] = TYPE_HEXAEDER;
                int i;
                for (i = 0; i < 8; i++)
                {
                    _vertices.push_back(vert[i] - 1);
                    _noOfVertices++;
                }
                _noOfElements++;
            }
        }
        if (strncmp(_c, "*node(", 6) == 0)
        { // new _line node
            int nodeNum;
            float xc, yc, zc;
            int iret = sscanf(_c + 6, "%d,%f,%f,%f", &nodeNum, &xc, &yc, &zc);
            if (iret != 4)
            {
                cerr << "parse error reading node " << iret << endl;
                break;
            }
            _xPoints[nodeNum - 1] = xc;
            _yPoints[nodeNum - 1] = yc;
            _zPoints[nodeNum - 1] = zc;
            _noOfPoints++;
        }
    }
    fclose(_fileHandle);
    debug();

    if (_noOfPoints == 0)
    {
        ERROR1("could not read points from file ", filename, _outputHandler);
    }
}

void MeshFileTransParserHmAscii::getLine(void)
{
    if (!_pushedBack)
    {
        if (fgets(_line, LINE_SIZE, _fileHandle) == NULL)
        {
            //   fprintf(stderr, "ReadFamu::getLine(): fgets failed\n" );
        }
    }
    else
    {
        _pushedBack = false;
    }
    _c = _line;
    while (*_c != '\0' && isspace(*_c))
    {
        _c++;
    }
}

void MeshFileTransParserHmAscii::pushBack(void)
{
    _pushedBack = true;
}

int MeshFileTransParserHmAscii::getMaxNodeNo(void) const
{
    int retval = (int)_xPoints.size(); // _xPoints[] is already extended to max. node number (filled with zeros where node no. is not defined)
    ASSERT0(retval >= _noOfPoints, "sorry, and internal error occured.", _outputHandler);
    return retval;
}

int MeshFileTransParserHmAscii::getInternalNodeNo(int) const
{
    ERROR0("illegal function call", _outputHandler);
    return 0;
}

void MeshFileTransParserHmAscii::debug(void) const
{
    //ArrayIo::writeArrayToDisk("_elements.txt", &_elements[0], _noOfElements);
    //ArrayIo::writeArrayToDisk("_vertices.txt", &_vertices[0], _noOfVertices);
    //ArrayIo::writeArrayToDisk("_types.txt",    &_types[0],    _noOfElements);
    //ArrayIo::writeArrayToDisk("_xPoints.txt",  &_xPoints[0],  _noOfPoints);
    //ArrayIo::writeArrayToDisk("_yPoints.txt",  &_yPoints[0],  _noOfPoints);
    //ArrayIo::writeArrayToDisk("_zPoints.txt",  &_zPoints[0],  _noOfPoints);
}
