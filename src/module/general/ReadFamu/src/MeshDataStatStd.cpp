/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataStatStd.h
 * a container for mesh data/binary file format.
 */

#include "MeshDataStatStd.h" // a container for mesh data/binary file format.
#include "errorinfo.h" // a container for error data.
#include "ReadFamu.h"
#include "Tools.h" // some helpful tools.

#define VERSION_COVISEMESH 0.1

MeshDataStatStd::~MeshDataStatStd()
{
    DELETE_ARRAY(_xPoints);
    DELETE_ARRAY(_yPoints);
    DELETE_ARRAY(_zPoints);
    DELETE_ARRAY(_vertices);
    DELETE_ARRAY(_elements);
    DELETE_ARRAY(_types);
    DELETE_ARRAY(_mesh2internalArr);
    _collectorNames.clear();
}

int MeshDataStatStd::getNoOfElements(void) const
{
    return _noOfElements;
}

int MeshDataStatStd::getNoOfPoints(void) const
{
    return _noOfPoints;
}

void MeshDataStatStd::getMeshData(
    int *noOfElements, int *noOfVertices, int *noOfPoints,
    int *elementsArr[], int *verticesArr[],
    float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
    int *typesArr[])
{
    *noOfElements = _noOfElements;
    *noOfVertices = _noOfVertices;
    *noOfPoints = _noOfPoints;

    *elementsArr = _elements;
    *verticesArr = _vertices;

    *xPointsArr = _xPoints;
    *yPointsArr = _yPoints;
    *zPointsArr = _zPoints;

    *typesArr = _types;
}

int MeshDataStatStd::getMaxNodeNo(void) const
{
    return _noOfPoints; // dogy, but this method should be decrepit when Ascii parser is improved, anyway.
}

// returns -1 if meshNodeNo is too high.
int MeshDataStatStd::getInternalNodeNo(int meshNodeNo) const
{
    int retval = -1;
    if (meshNodeNo > _maxNodeNoMesh)
    {
        retval = -1;
    }
    else
    {
        retval = _mesh2internalArr[meshNodeNo];
        ASSERT(retval >= -1 && retval < 1e8, _outputHandler);
    }
    return retval;
}

void MeshDataStatStd::debug(void) const
{
    ArrayIo::writeArrayToDisk("_elements.txt", _elements, _noOfElements);
    ArrayIo::writeArrayToDisk("_vertices.txt", _vertices, _noOfVertices);
    ArrayIo::writeArrayToDisk("_types.txt", _types, _noOfElements);
    ArrayIo::writeArrayToDisk("_xPoints.txt", _xPoints, _noOfPoints);
    ArrayIo::writeArrayToDisk("_yPoints.txt", _yPoints, _noOfPoints);
    ArrayIo::writeArrayToDisk("_zPoints.txt", _zPoints, _noOfPoints);
}
