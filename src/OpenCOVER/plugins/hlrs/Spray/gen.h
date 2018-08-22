/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GEN_H
#define GEN_H

#include "types.h"
#include "nozzle.h"
#include "parser.h"
#include "raytracer.h"

#include <PluginUtil/coSphere.h>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Quat>
#include <osg/Vec3>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRTui.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cmath>

using namespace covise;
using namespace opencover;

const float g = 9.81;
const float Pi = 3.141592653;

class gen
{
private:
    coSphere* coSphere_;
    osg::Geode* geode_;
    osg::Vec4 currentColor = osg::Vec4(1,1,1,1);
    osg::Vec3 gravity = osg::Vec3(0,0,g);
    osg::ref_ptr<osg::MatrixTransform> transform_;

    bool outOfBound = false;

    float tCur = 1;


    std::string cwModelType = "STOKES";

    float densityOfFluid = 1.18;
    int reynoldsThreshold = 170000;
    float nu = 0.0171;
    float cwLam = 0.45;
    float cwTurb = 0.15;
    float minimum = 0.000025;
    float deviation = 0.00005;
    int iterations = 4;

    bool* particleOutOfBound;
    bool* firstHit;
    float* prevHitDis;
    int* prevHitDisCounter;

    int outOfBoundCounter = 0;

    particleParam p;

    void updateCoSphere();
    float reynoldsNr(float v, double d);

protected:
    float* x, *y, *z, *vx, *vy, *vz;
    double *r, *m;
    float densityOfParticle = 1000;
    class nozzle* owner_;
    int particleCount_ = 1000;
    float initPressure_;

    void setCoSphere();

public:
    gen(float pInit, class nozzle *owner);
    virtual ~gen();

    void init();    

    void setColor(osg::Vec4 newColor){
        currentColor = newColor;
        for(int i = 0; i< particleCount_;i++){
            coSphere_->setColor(i, newColor.x(), newColor.y(), newColor.z(), newColor.w());
        }
    }

    osg::Geode* getGeode(){
        return geode_;
    }

    bool isOutOfBound(){
        return outOfBound;
    }

    void updatePos(osg::Vec3 boundingBox);

    float getInitPressure()
    {
        return initPressure_;
    }

    float getTimeStep()
    {
        return tCur;
    }

    float getMinimum()
    {
        return minimum;
    }

    float getDeviation()
    {
        return deviation;
    }

    void setMinimum(float newMinimum)
    {
        minimum = newMinimum;
    }

    void setDeviation(float newDeviation)
    {
        deviation = newDeviation;
    }
};



class imageGen : public gen
{
private:
    pImageBuffer* iBuf_;
public:
    imageGen(pImageBuffer* iBuf, float pInit, class nozzle* owner);
    ~imageGen();

    void seed();
};





class standardGen : public gen
{
private:
    float sprayAngle_ = 0;
    std::string decoy_;

public:
    standardGen(float sprayAngle, std::string decoy, float pInit, class nozzle *owner);

    void seed();
};

#endif // GEN_H
