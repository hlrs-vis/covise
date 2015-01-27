/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshDataStatBinary.h
 * a container for mesh data/binary file format.
 * FAMU Copyright (C) 1998-2005 Institute for Theory of Electrical Engineering
 */

//#include "MeshDataStatBinary.h"  // a container for mesh data/binary file format.

#ifndef __MeshDataStatBinary_h__
#define __MeshDataStatBinary_h__

#include "objectinputstream.hxx" // a stream that can be used for deserialization.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "MeshDataStat.h" // a container for mesh data.
#include <vector>
#include <string>

/**
 * a container for mesh data/binary file format.
 */
class MeshDataStatBinary : public MeshDataStat
{
public:
    MeshDataStatBinary(ObjectInputStream *archive,
                       OutputHandler *outputHandler);
    MeshDataStatBinary(int noOfElements, int noOfVertices, int noOfPoints,
                       int elementsArr[], int verticesArr[],
                       float xPointsArr[], float yPointsArr[], float zPointsArr[],
                       int typesArr[],
                       int mesh2internalArr[], int maxNodeNoMesh,
                       OutputHandler *outputHandler);
    virtual ~MeshDataStatBinary();

    virtual int getNoOfElements(void) const;
    virtual int getNoOfPoints(void) const;
    virtual void getMeshData(int *noOfElements, int *noOfVertices, int *noOfPoints,
                             int *elementsArr[], int *verticesArr[],
                             float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
                             int *typesArr[]);
    int getInternalNodeNo(int meshNodeNo) const;
    virtual int getMaxNodeNo(void) const;

private:
    OutputHandler *_outputHandler;

    int _noOfPoints;
    float *_xPoints;
    float *_yPoints;
    float *_zPoints;

    int *_mesh2internalArr;
    int _maxNodeNoMesh;

    int _noOfVertices;
    int *_vertices;

    int _noOfElements;
    int *_elements;

    int *_types;

    std::vector<int> _collectorNumbers;
    std::vector<std::string> _collectorNames;

    void readArr_INT2int(int *arr[], int *size, ObjectInputStream *archive) const;
    void readArr_CHAR2int(int *arr[], int *size, ObjectInputStream *archive) const;

    void debug(void) const;
};

#endif
