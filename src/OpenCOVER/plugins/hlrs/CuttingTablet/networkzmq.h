/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NETWORK_H
#define NETWORK_H

#include <map>
#include <vector>

#include <cstdlib>
#include <iostream>
#include <zmq.h>
#include <boost/signals2/mutex.hpp>

#include <osg/Vec3>
#include <osg/Matrix>

class Mesh;

class Server
{

public:
    Server(int portMatrix, int portMesh);

    void setMarkerPosition(const int id,
                           const float px, const float py, const float pz);

    void setMarkerMatrix(const int id,
                         const osg::Matrix &matrix);

    void setPosition(const int id,
                     const float px, const float py, const float pz);
    void setNormal(const int id,
                   const float nx, const float ny, const float nz);

    void removePosition(const int id);
    void removeNormal(const int id);
    void sendGeometry(const int numVertices, const float *vertices,
                      const int numIndices, const unsigned int *indices,
                      const int numTexCoords, const float *texCoords);

    osg::Vec3 getNormal() const;
    osg::Vec3 getPosition() const;

    void poll();
    int getDataSet();

private:
    enum TYPE
    {
        MSG_MARKER = 1,
        MSG_MESH = 42
    };

    std::map<int, osg::Vec3> position;
    std::map<int, osg::Vec3> normal;

    std::map<int, osg::Vec3> markerPosition;

    void *context;
    void *socketMatrix;
    void *socketMesh;

    boost::signals2::mutex mutex;
    Mesh *mesh;
    int num;
    int dataSet;
};

#endif
