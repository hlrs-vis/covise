#pragma once

#include<vector>
#include<memory>

#include<osg/Matrix>
#include<osg/Geode>
#include<osg/Geometry>
#include<osg/MatrixTransform>

#include <OpenVRUI/osg/mathUtils.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coVRPluginSupport.h>

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
    SensorPosition(osg::Matrix matrix, bool visible);
    virtual ~SensorPosition(){}; //do I now need to implement move ....
    
    virtual bool preFrame();
    virtual osg::Geode* draw() = 0;

    //virtual const Orientation* getRandomOrientation()const;
    virtual void calcVisibility();

    virtual void setMatrix(osg::Matrix matrix);
    void setVisibilityMatrix(VisibilityMatrix<float>&& visMat){m_CurrentOrientation.setVisibilityMatrix(std::move(visMat));}
    void setCurrentOrientation(Orientation);

    virtual int getNbrOfOrientations()const{return 1;}
    virtual void VisualizationVisible(bool status)const;

    osg::ref_ptr<osg::Group> getSensor()const;
    osg::Matrix getMatrix()const;
    const VisibilityMatrix<float>& getVisibilityMatrix()const{return m_CurrentOrientation.getVisibilityMatrix();}
    virtual Orientation* getSpecificOrientation(int position){return &m_CurrentOrientation;}


protected:    
    virtual double calcRangeDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcWidthDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcHeightDistortionFactor(const osg::Vec3& point)const = 0;

    virtual void showOriginalSensorSize() = 0;
    virtual void showIconSensorSize() = 0;

    virtual VisibilityMatrix<float> calcVisibilityMatrix(coCoord& euler) = 0;
    void checkForObstacles();
    
    const unsigned int m_NodeMask = UINT32_MAX & ~opencover::Isect::Intersection & ~opencover::Isect::Pick;
    std::unique_ptr<opencover::coVR3DTransRotInteractor> m_Interactor; 
    Orientation m_CurrentOrientation;                                       // Visualized orientation
    VisibilityMatrix<float> m_VisMatSensorPos;                              // VisibilityMatrix(only contains intersections with obstacles)
    int m_Scale{10};                                                        // Scale factor for sensor visualization
    osg::ref_ptr<osg::Group> m_SensorGroup;
    osg::ref_ptr<osg::MatrixTransform> m_SensorMatrix;

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
    SensorWithMultipleOrientations(osg::Matrix matrix);
    ~SensorWithMultipleOrientations() override{};
    
    bool preFrame() override;  
    void calcVisibility() override;
    int getNbrOfOrientations()const override{return m_Orientations.size();}

    void setMatrix(osg::Matrix matrix)override; // --> TODO: anpassen !
    Orientation* getSpecificOrientation(int position)override{return &m_Orientations.at(position);}
    std::vector<Orientation>& getOrientations(){return m_Orientations;}

protected:
    virtual osg::Geode* drawOrientation() = 0;

    void createSensorOrientations();
    void deleteSensorOrientations();

    virtual bool compareOrientations(const Orientation& lhs, const Orientation& rhs);

    std::vector<Orientation> m_Orientations;
    osg::ref_ptr<osg::Group> m_OrientationsGroup;


private:
    void decideWhichOrientationsAreRequired(const Orientation&& orientation);
    void replaceOrientationWithLastElement(int index);
    bool isVisibilityMatrixEmpty(const Orientation& orientation);

    
    SensorProps m_SensorProps;
};



