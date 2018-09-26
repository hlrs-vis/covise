/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TYPES_H
#define TYPES_H

#include <osg/Vec3>

#define HIT_X_POS 1
#define HIT_Y_POS 2
#define HIT_Z_POS 3
#define HIT_X_NEG 4
#define HIT_Y_NEG 5
#define HIT_Z_NEG 6
#define NOT_HIT 0
#define OUT_OF_BOUND -1

typedef struct{
    float* dataBuffer;
    int samplingPoints;
    int centerX;
    int centerY;
    int numOfEntries;
}pImageBuffer;

class particle
{
public:
    osg::Vec3 pos;
    osg::Vec3 velocity;
    double r;
    double m;
    float time;
    bool particleOutOfBound = false;
    bool firstHit = false;
};

#endif // TYPES_H
