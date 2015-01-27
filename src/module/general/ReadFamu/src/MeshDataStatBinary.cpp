/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataStatBinary.h
 * a container for mesh data/binary file format.
 */

#include "MeshDataStatBinary.h" // a container for mesh data/binary file format.
#include "errorinfo.h" // a container for error data.
#include "ReadFamu.h"
#include "Tools.h" // some helpful tools.

#define VERSION_COVISEMESH 0.1

MeshDataStatBinary::MeshDataStatBinary(ObjectInputStream *archive,
                                       OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfPoints(0)
    , _xPoints(0)
    , _yPoints(0)
    , _zPoints(0)
    , _mesh2internalArr(0)
    , _maxNodeNoMesh(0)
    , _noOfVertices(0)
    , _vertices(0)
    , _noOfElements(0)
    , _elements(0)
    , _types(0)
{
    double version = archive->readDouble();
    ASSERT(version == VERSION_COVISEMESH, _outputHandler);

    int noOfCollectors = archive->readINT();
    int i;
    for (i = 0; i < noOfCollectors; i++)
    {
        _collectorNumbers.push_back((int)archive->readINT());
        _collectorNames.push_back(archive->readString());
        char buf[1000];
        sprintf(buf, "component %d %s\n", _collectorNumbers[i], _collectorNames[i].c_str());
        _outputHandler->displayString(buf);
    }

    readArr_INT2int(&_elements, &_noOfElements, archive);
    readArr_CHAR2int(&_types, &_noOfElements, archive);
    readArr_INT2int(&_vertices, &_noOfVertices, archive);
    UINT_F arrSize = 0;
    archive->readArray(&_xPoints, (UINT_F *)&arrSize);
    _noOfPoints = (int)arrSize;
    archive->readArray(&_yPoints, (UINT_F *)&arrSize);
    _noOfPoints = (int)arrSize;
    archive->readArray(&_zPoints, (UINT_F *)&arrSize);
    _noOfPoints = (int)arrSize;
    readArr_INT2int(&_mesh2internalArr, &_maxNodeNoMesh, archive);
    _maxNodeNoMesh--; // because array starts with 1
    debug();
}

MeshDataStatBinary::MeshDataStatBinary(
    int noOfElements, int noOfVertices, int noOfPoints,
    int elementsArr[], int verticesArr[],
    float xPointsArr[], float yPointsArr[], float zPointsArr[],
    int typesArr[],
    int mesh2internalArr[], int maxNodeNoMesh,
    OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfPoints(noOfPoints)
    , _xPoints(xPointsArr)
    , _yPoints(yPointsArr)
    , _zPoints(zPointsArr)
    , _mesh2internalArr(mesh2internalArr)
    , _maxNodeNoMesh(maxNodeNoMesh)
    , _noOfVertices(noOfVertices)
    , _vertices(verticesArr)
    , _noOfElements(noOfElements)
    , _elements(elementsArr)
    , _types(typesArr)
{
}

MeshDataStatBinary::~MeshDataStatBinary()
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

void MeshDataStatBinary::readArr_CHAR2int(int *arr[],
                                          int *size,
                                          ObjectInputStream *archive) const
{
    UINT_F size_f = 0;
    CHAR_F *arr_f = NULL;
    archive->readArray(&arr_f, &size_f);
    *size = (int)size_f;

    *arr = new int[*size];
    int i;
    for (i = 0; i < *size; i++)
    {
        (*arr)[i] = (int)arr_f[i];
    }
    delete[] arr_f;
}

void MeshDataStatBinary::readArr_INT2int(int *arr[],
                                         int *size,
                                         ObjectInputStream *archive) const
{
    UINT_F size_f = 0;
    INT_F *arr_f = NULL;
    archive->readArray(&arr_f, &size_f);
    *size = (int)size_f;

    *arr = new int[*size];
    int i;
    for (i = 0; i < *size; i++)
    {
        (*arr)[i] = (int)arr_f[i];
    }
    delete[] arr_f;
}

int MeshDataStatBinary::getNoOfElements(void) const
{
    return _noOfElements;
}

int MeshDataStatBinary::getNoOfPoints(void) const
{
    return _noOfPoints;
}

void MeshDataStatBinary::getMeshData(
    int *noOfElements, int *noOfVertices, int *noOfPoints,
    int *elementsArr[], int *verticesArr[],
    float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
    int *typesArr[])
{
#if 1 // debugging
    std::ostringstream s;
    s << "no. of elements: " << _noOfElements << ", ";
    s << "no. of vertices: " << _noOfVertices;
    std::string ss = s.str();
    _outputHandler->displayString(ss.c_str());
#endif

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

int MeshDataStatBinary::getMaxNodeNo(void) const
{
    return _noOfPoints; // dogy, but this method should be decrepit when Ascii parser is improved, anyway.
}

// returns -1 if meshNodeNo is too high.
int MeshDataStatBinary::getInternalNodeNo(int meshNodeNo) const
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

void MeshDataStatBinary::debug(void) const
{
    /*ArrayIo::writeArrayToDisk("_elements.txt", _elements, _noOfElements);
    ArrayIo::writeArrayToDisk("_vertices.txt", _vertices, _noOfVertices);
    ArrayIo::writeArrayToDisk("_types.txt",    _types,    _noOfElements);
    ArrayIo::writeArrayToDisk("_xPoints.txt",  _xPoints,  _noOfPoints);
    ArrayIo::writeArrayToDisk("_yPoints.txt",  _yPoints,  _noOfPoints);
    ArrayIo::writeArrayToDisk("_zPoints.txt",  _zPoints,  _noOfPoints);
    ArrayIo::writeArrayToDisk("_mesh2internalArr.txt",  _mesh2internalArr, _maxNodeNoMesh);*/
}
