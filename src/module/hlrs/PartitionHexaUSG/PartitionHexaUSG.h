/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PARTITIONHEXAUSG_H
#define PARTITIONHEXAUSG_H

/** 
 * PartitionHexaUSG uses METIS to partition an USG and associated data objects.
 * The USG must consist of hexahedronal elements exclusively.
 *
 * @author Florian Niebling
 * @date 10.04.2010
 */

#ifdef WIN32
#include <hash_map>
#include <hash_set>
#else
#include <ext/hash_map>
#include <ext/hash_set>
#endif

#include <api/coSimpleModule.h>

#define DATA_PORTS 4

using namespace covise;

class Partition;

class PartitionHexaUSG : public coSimpleModule
{
public:
    enum
    {
        NONE = 0,
        PER_VERTEX,
        PER_ELEMENT
    };
    PartitionHexaUSG(int argc, char *argv[]);

private:
    virtual int compute(const char *port);

    coInputPort *p_gridIn;
    coOutputPort *p_gridOut;

    coInputPort *p_scalarIn[4];
    coOutputPort *p_scalarOut[4];

    coInputPort *p_vectorIn[4];
    coOutputPort *p_vectorOut[4];

    coIntScalarParam *p_num;

    std::vector<Partition *> partitions;
};

/**
 * Representation of a partition of a hexahedronal USG and various associated
 * scalar data objects
 */
class Partition
{

public:
    Partition();

    /**
    * Create an empty COVISE USG with numElems hexahedron elements
    *
    * @param name the name of the COVISE object
    */
    void createUSG(const char *name);

    /**
    * Create an empty scalar float data object with either per vertex or
    * per element size regarding the USG
    *
    * @param index the number of the scalar data field
    * @param name the name of the COVISE object
    * @param dataBinding PER_VERTEX or PER_ELEMENT
    */
    void createScalarData(int index, const char *name, int dataBinding);

    /**
    * Create an empty vector float data object with either per vertex or
    * per element size regarding the USG
    *
    * @param index the number of the vector data field
    * @param name the name of the COVISE object
    * @param dataBinding PER_VERTEX or PER_ELEMENT
    */
    void createVectorData(int index, const char *name, int dataBinding);

    /**
    * add a mapping from the index of a coordinate point in the complete USG to
    * the index of the point in the partition
    *
    * @param point the index of the coordinate in the complete USG
    * @param map the index of the coordinate in the partition
    */
    void setPointMapping(int point, int map);

    /**
    * Get the index of a coordinate point in the partition given the
    * index in the complete USG
    *
    * @param point the index in the complete USG
    * @return the index mapping or -1 if the point is not yet in the partition
    */
    int getPointMapping(int point);

    /**
    * Set a coordinate point in the partition
    *
    * @param offset the index in the coordinate field
    * @param xp the x-coordinate of the point
    * @param yp the y-coordinate of the point
    * @param zp the z-coordinate of the point
    */
    void setXYZ(int offset, float xp, float yp, float zp);

    /**
    * Set a single scalar data element in a coDoFloat data object
    *
    * @param index the number of the scalar data field
    * @param offset the offset into the field
    * @param data the value of the data element
    */
    void setScalarData(int index, int offset, float data);

    /**
    * Set a single vector data element in a coDoVec3 data object
    *
    * @param index the number of the vector data field
    * @param offset the offset into the field
    * @param u the u-value of the data element
    * @param v the v-value of the data element
    * @param w the w-value of the data element
    */
    void setVectorData(int index, int offset, float u, float v, float w);

    // USG members
    int numElems;
#ifdef WIN32
    stdext::hash_set<int> numCoords;
#else
    __gnu_cxx::hash_set<int> numCoords;
#endif
    int *elem, *conn, *type;

    // COVISE grid and data objects
    coDoUnstructuredGrid *gridObject;
    coDoFloat *scalarDataObject[DATA_PORTS];
    coDoVec3 *vectorDataObject[DATA_PORTS];

private:
/**
    * Mapping from coordinate numbers in the incoming USG to
    * coordinate numbers in the partition
    */
#ifdef WIN32
    stdext::hash_map<int, int> pointMap;
#else
    __gnu_cxx::hash_map<int, int> pointMap;
#endif

    // COVISE data object pointers
    float *x, *y, *z;
    float *scalarData[DATA_PORTS];
    float *vectorDataU[DATA_PORTS];
    float *vectorDataV[DATA_PORTS];
    float *vectorDataW[DATA_PORTS];
};

#endif
