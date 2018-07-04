#ifndef GEN_H
#define GEN_H

#include "../../../PluginUtil/coSphere.h"
#include "../../../../../../../../usr/include/osg/MatrixTransform"
#include "../../../../../../../../usr/include/osg/Matrix"
#include "../../../../../../../../usr/include/osg/ShapeDrawable"
#include "../../../../../../../../usr/include/osg/Shape"
#include "../../../../../../../../usr/include/osg/Quat"
#include "../../../../../../../../usr/include/osg/Vec3f"

#include "../../../../kernel/config/CoviseConfig.h"
using namespace covise;

#include "../../../../OpenCOVER/cover/coVRPluginSupport.h"
#include "../../../../OpenCOVER/cover/coVRFileManager.h"
#include "../../../../OpenCOVER/cover/coVRPlugin.h"
#include "../../../../OpenCOVER/cover/coVRTui.h"
#include "../../../../OpenCOVER/PluginUtil/coSphere.h"
#include "../../../../OpenCOVER/PluginUtil/coVR3DTransRotInteractor.h"
#include "types.h"
#include "nozzle.h"
#include "math.h"

#include "parser.h"
#include "raytracer.h"
using namespace opencover;

const float g = 9.81;
const float Pi = 3.141592653;

class gen
{
private:
    coSphere* coSphere_;
    /*osg::ref_ptr<osg::Geode*> */osg::Geode* geode_;
    osg::Vec4 currentColor = osg::Vec4(1,1,1,1);
    osg::ref_ptr<osg::MatrixTransform> transform_;

    bool outOfBound = false;
    bool isHit_ = false;

    float t = 0;
    float tCur = 0.001;
    float initPressure_ = 2;

    std::string cwModelType = "STOKES";
    float densityOfFluid = 1.18;
    int reynoldsThreshold = 170000;
    float nu = 0.0000171;
    float cwLam = 0.45;
    float cwTurb = 0.15;

    bool* particleOutOfBound;
    bool* firstHit;
    int outOfBoundCounter = 0;

    int sphereHitCounter = 0;

    particleParam p;

    void updateCoSphere();
    float reynoldsNr(float v, float d);

protected:
    float* x, *y, *z, *vx, *vy, *vz, *r, *m;
    class nozzle* owner_;
    int particleCount_ = 1000;
    void init();
public:
    gen(float pInit, class nozzle *owner);
    virtual ~gen();


    osg::Vec3 gravity = osg::Vec3(0,0,g*1000);

    void setColor(osg::Vec4 newColor){
        currentColor = newColor;
        for(int i = 0; i< particleCount_;i++){
            coSphere_->setColor(i, newColor.x(), newColor.y(), newColor.z(), newColor.w());
        }
    }

    osg::Vec4 getColor(){
        return currentColor;
    }

    /*osg::ref_ptr<osg::Geode*>*/ osg::Geode* getGeode(){
        return geode_;
    }

    bool isOutOfBound(){
        return outOfBound;
    }

    void setHit(bool hit){
        isHit_ = hit;
    }

    bool isHit(){
        return isHit_;
    }

    void updatePos(osg::Vec3 boundingBox);

    void setCoSphere();

    float getInitPressure()
    {
        return initPressure_;
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
    const char* decoy_;

public:
    standardGen(float sprayAngle, const char *decoy, float pInit, class nozzle *owner);

    void seed();
};

#endif // GEN_H
