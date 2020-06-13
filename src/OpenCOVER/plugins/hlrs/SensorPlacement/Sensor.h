#pragma once

#include<vector>
#include<memory>

#include<osg/Matrix>
#include<osg/Geode>
#include<osg/Geometry>
#include<osg/MatrixTransform>

#include <OpenVRUI/osg/mathUtils.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>

double calcValueInRange(double oldMin, double oldMax, double newMin, double newMax, double oldValue);

template<typename T>
using VisibilityMatrix = std::vector<T>;

class Orientation
{
public:

    Orientation(osg::Matrix matrix,VisibilityMatrix<float>&& visMat);
    Orientation(coCoord euler,VisibilityMatrix<float>&& visMat);
    Orientation(osg::Matrix);
    //bool operator >> (const Orientation& other) const;

    const osg::Matrix& getMatrix()const{return m_Matrix;}
    const coCoord& getEuler()const{return m_Euler;}
    const VisibilityMatrix<float>& getVisibilityMatrix()const{return m_VisibilityMatrix;}

    void setMatrix(osg::Matrix matrix);
    void setMatrix(coCoord euler);
    void setVisibilityMatrix(VisibilityMatrix<float>&& visMat);

private:
    osg::Matrix m_Matrix;
    coCoord m_Euler;
    VisibilityMatrix<float> m_VisibilityMatrix;

};

class SensorVisualization;
class SensorPosition
{
public:
    SensorPosition(osg::Matrix matrix);
    virtual ~SensorPosition(){}; //do I now need to implement move ....
    
    bool preFrame();
    virtual VisibilityMatrix<float> calcVisibilityMatrix(coCoord& euler) = 0;
    virtual osg::ref_ptr<osg::Group> getSensor()const;

    virtual void setMatrix(osg::Matrix matrix);
    void checkForObstacles();

    osg::Matrix getMatrix()const;
    SensorVisualization& getSensorVisualization(){return *m_SensorVisualization;}
    const VisibilityMatrix<float>& getVisibilityMatrix()const{return m_Orientation.getVisibilityMatrix();}

    void setVisibilityMatrix(VisibilityMatrix<float>&& visMat){m_Orientation.setVisibilityMatrix(std::move(visMat));}
protected:    
    virtual double calcRangeDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcWidthDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcHeightDistortionFactor(const osg::Vec3& point)const = 0;

    Orientation m_Orientation; //each Sensor has at least one Orientation / Position 
    VisibilityMatrix<float> m_VisMatSensorPos; 
    std::unique_ptr<SensorVisualization> m_SensorVisualization;
};
struct SensorProps
{
    //Step sizes for the Orientations in Degree
    int m_StepSizeX{10};
    int m_StepSizeY{45};
    int m_StepSizeZ{5};
};

class SensorWithMultipleOrientations : public SensorPosition
{
public:
    ~SensorWithMultipleOrientations() override{};
    SensorWithMultipleOrientations(osg::Matrix matrix):SensorPosition(matrix){};
    void setMatrix(osg::Matrix matrix)override; // --> TODO: anpassen !
protected:
    std::vector<Orientation> m_Orientations;
    void createSensorOrientations();

    virtual bool compareOrientations(const Orientation& lhs, const Orientation& rhs);
private:
    SensorProps m_SensorProps;

    void decideWhichOrientationsAreRequired(const Orientation&& orientation);
    void replaceOrientationWithLastElement(int index);
};

class SensorVisualization
{
public:
    SensorVisualization(SensorPosition* sensor);
    virtual bool preFrame();
    virtual osg::Geode* draw(){};
    void setMatrix(osg::Matrix matrix){m_Matrix->setMatrix(matrix);}
    osg::Matrix getMatrix()const{return m_Matrix->getMatrix();}
    virtual osg::ref_ptr<osg::Group> getSensorVisualization()const{return m_Group.get();}
protected:
    int m_Scale{10};
    SensorPosition* m_Sensor;  
    std::unique_ptr<opencover::coVRIntersectionInteractor> m_Interactor; 
    osg::ref_ptr<osg::Group> m_Group;
    osg::ref_ptr<osg::MatrixTransform> m_Matrix;

    virtual void showOriginalSensorSize() = 0;
    virtual void showIconSensorSize() = 0;

};


