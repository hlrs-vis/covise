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

#include <osg/Array>

namespace osg
{
class Geode;
class Geometry;
class MatrixTransform;
class StateSet;
}

namespace opencover
{
#define NUM_SAMPLES 200
class COVEREXPORT coStatGraph
{
public:
    coStatGraph(int col);
    ~coStatGraph();
    float maxValue;
    void updateValues();
    void updateMinMax(double val);
    osg::MatrixTransform *graph;
    float max;
    float sum;

private:
    float zCoords[NUM_SAMPLES];
    osg::StateSet *myGeostate;
    osg::Vec3Array *coord;
    osg::Vec3Array *normal;
    ushort *coordIndex;
    osg::ShortArray *normalIndex;
    osg::Geometry *geoset1;
    osg::Geode *geode;
};

class COVEREXPORT coClusterStat
{
public:
    coClusterStat(int id);
    ~coClusterStat();
    coStatGraph *renderGraph;
    coStatGraph *sendGraph;
    coStatGraph *recvGraph;
    float renderMax;
    float sendMax;
    float recvMax;
    void show();
    void hide();
    void updateMinMax(double renderTime, double netSend, double netRecv);
    void updateValues();

private:
    osg::MatrixTransform *graph;
    int ID;
};
}
#endif
