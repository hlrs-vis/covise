/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CLUSTER_STAT_H
#define CO_CLUSTER_STAT_H

/*! \file
 \brief  timing statistics for cluster rendering

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2005
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coTypes.h>

#include <vsg/core/Array.h>
#include <vsg/nodes/MatrixTransform.h>

namespace osg
{
class Geode;
class Geometry;
class MatrixTransform;
class StateSet;
}

namespace vive
{
#define NUM_SAMPLES 200
class VVCORE_EXPORT vvStatGraph
{
public:
    vvStatGraph(int col);
    ~vvStatGraph();
    float maxValue;
    void updateValues();
    void updateMinMax(double val);
    vsg::MatrixTransform *graph;
    float max;
    float sum;

private:
    float zCoords[NUM_SAMPLES];
    vsg::vec3Array *coord;
    vsg::vec3Array *normal;
    ushort *coordIndex;
    vsg::svec3Array *normalIndex;
    vsg::Node *geoset1;
};

class VVCORE_EXPORT coClusterStat
{
public:
    coClusterStat(int id);
    ~coClusterStat();
    vvStatGraph *renderGraph;
    vvStatGraph *sendGraph;
    vvStatGraph *recvGraph;
    float renderMax;
    float sendMax;
    float recvMax;
    void show();
    void hide();
    void updateMinMax(double renderTime, double netSend, double netRecv);
    void updateValues();

private:
    vsg::MatrixTransform *graph;
    int ID;
};
}
#endif
