#ifndef NOZZLE_H
#define NOZZLE_H

namespace opencover
{
class coVR3DTransRotInteractor;
}
using namespace opencover;

#include <string>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include "../../../../OpenCOVER/cover/coVRTui.h"
#include "../../../../../../../../usr/include/osgDB/ReadFile"
#include <osg/Matrix>
#include <osg/Node>
#include <osg/ShapeDrawable>
#include <osg/BoundingBox>

#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

#include "gen.h"
#include "types.h"

#include "parser.h"

#define posToCount(x,y,s) (3*x+y*s)
#define getMin(x,y) x<y?x:y

class nozzle : public coVR3DTransRotInteractor
{
private:
    int counter = 0;
    int nozzleID = 0;

    int prevGenCreate = 0;

    float initPressure_ = 4;

    //osg::ref_ptr<osg::Geode*> geode_;
    osg::ref_ptr<osg::MatrixTransform> transform_;
    osg::Geode* geode_;
    osg::Vec3 boundingBox_ = osg::Vec3(2000,2000,2000);
    osg::Cylinder* cylinder_;
    osg::ShapeDrawable* shapeDrawable_;
    osg::Vec4 nozzleColor = osg::Vec4(1,1,1,1);
    osg::Vec4 currentColor_ = osg::Vec4(1,1,0,1);

    void updateColor();
    void deleteGen(class gen* current);

    bool initialized = false;
    bool labelRegistered = false;

    ui::Label* nozzleLabel_;


protected:
    void createGeometry();
    int particleCount_ = 1000;
    std::string nozzleName_;
    std::list<class gen*> genList;

    bool failed = false;
public:

    nozzle(osg::Matrix initialMat, float size, std::string nozzleName);
    virtual ~nozzle();

    virtual void createGen();
    void updateGen();

    void setColor(osg::Vec4 newColor){
        currentColor_ = newColor;
        updateColor();
    }

    osg::Vec4 getColor(){
        return currentColor_;
    }

    void setInitPressure(float newPressure)
    {
        initPressure_ = newPressure;
    }

    float getInitPressure()
    {
        return initPressure_;
    }

    void resizeBoundingBox(osg::Vec3 newSize){
        boundingBox_ = newSize;
    }

    virtual void save(std::string pathName, std::string fileName);

    void setID(int ID);
    int getID();

    std::string getName()
    {
        return nozzleName_;

    }

    ui::Label* registerLabel()
    {
        labelRegistered = true;
        return nozzleLabel_;
    }

    bool isRegistered()
    {
        return labelRegistered;
    }

    ui::Label* getLabel()
    {
        return nozzleLabel_;
    }

    void setNozzleColor(osg::Vec4 newColor)
    {
        nozzleColor = newColor;
        shapeDrawable_->setColor(nozzleColor);
    }

    bool isFailed()
    {
        return failed;
    }

    void keepSize();

    osg::MatrixTransform* getMatrixTransform()
    {
        return transform_->asMatrixTransform();
    }

};



class standardNozzle : public nozzle
{
private:
    float sprayAngle_ = 0;
    const char* decoy_;
public:
    const char* type = "standard";
    standardNozzle(float sprayAngle, std::__cxx11::string decoy, osg::Matrix initialMat, float size, std::string nozzleName);

    void createGen();
    void save(std::string pathName, std::string fileName);

    float getSprayAngle(){
        return sprayAngle_;
    }

    const char* getDecoy(){
        return decoy_;
    }
};

class imageNozzle : public nozzle
{
private:
    std::string fileName_;
    std::string pathName_;
    std::string imageFilePath;

    pImageBuffer iBuf;

    int samplingPoints = 1000;

    float pixel_to_mm_;
    float pixel_to_flow_;
    float pixel_to_radius_;
    int colorDepth_;
    int colorThreshold_ = 100;


    bool square = false;
    bool circle = true;
    bool readImage();
public:
    const char* type = "image";

    imageNozzle(std::string pathName, std::string fileName, osg::Matrix initialMat, float size, std::string nozzleName);
    ~imageNozzle();

    void createGen();
    void save(std::string pathName, std::string fileName);
    pImageBuffer getPImageBuffer()
    {
        return iBuf;
    }

};

#endif // NOZZLE_H
