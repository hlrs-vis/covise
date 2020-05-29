#pragma once

#include<vector>
#include<memory>

#include<osg/Matrix>
#include<osg/Geode>
#include<osg/Geometry>
#include<osg/MatrixTransform>

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <OpenVRUI/osg/mathUtils.h>

std::vector<osg::Vec3> Vec2DimToVec(std::vector<std::vector<int>> input);


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


class SensorPosition
{
public:
    SensorPosition(osg::Matrix matrix);
    virtual ~SensorPosition(){}; //do I now need to implement move ....
    
    virtual bool preFrame() = 0;
    virtual VisibilityMatrix<float> calcVisibilityMatrix(coCoord& euler) = 0;
    virtual osg::ref_ptr<osg::Group> getSensor() = 0;

    void checkForObstacles();

protected:    
    virtual double calcRangeDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcWidthDistortionFactor(const osg::Vec3& point)const = 0;
    virtual double calcHeightDistortionFactor(const osg::Vec3& point)const = 0;

    Orientation m_Orientation; //each Sensor has at least one Orientation / Position 
    VisibilityMatrix<float> m_VisMatSensorPos; 

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
protected:
    std::vector<Orientation> m_Orientations;
    void createSensorOrientations();

    virtual bool compareOrientations(const Orientation& lhs, const Orientation& rhs);
private:
    SensorProps m_SensorProps;

    void decideWhichOrientationsAreRequired(const Orientation&& orientation);
    void replaceOrientationWithLastElement(int index);
};

struct CameraProps
{  
private:
    // Full HD Camera
    float m_ImageHeightPixel{1080};
    float m_ImageWidthPixel{1920};

public:
    float m_FoV{60.f};
    float m_DepthView{40.f};
 
    float m_ImgWidth{2*m_DepthView*std::tan(m_FoV/2*(float)osg::PI/180)};
    float m_ImgHeight{m_ImgWidth / (m_ImageWidthPixel/m_ImageHeightPixel)};

    void updateFoV(float fov);
    void updateDepthView(float dof);

    int getImageHeightPixel()const{return m_ImageHeightPixel;}
    int getImageWidthPixel()const{return m_ImageWidthPixel;}

};

class Camera : public SensorWithMultipleOrientations
{
public:
    Camera(osg::Matrix matrix);
    ~Camera()override{};
    
    bool preFrame() override;
    VisibilityMatrix<float> calcVisibilityMatrix(coCoord& euler) override;
    osg::ref_ptr<osg::Group> getSensor() override{return m_CameraMatrix;}

protected:
    double calcRangeDistortionFactor(const osg::Vec3& point)const override;
    double calcWidthDistortionFactor(const osg::Vec3& point)const override;
    double calcHeightDistortionFactor(const osg::Vec3& point)const override;

private:
    CameraProps m_CameraProps;
    std::unique_ptr<opencover::coVR3DTransRotInteractor> m_Interactor;

    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Vec4Array> m_Colors;
    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Geometry> m_Geometry;
    osg::ref_ptr<osg::MatrixTransform> m_CameraMatrix;

    std::vector<osg::ref_ptr<osg::MatrixTransform>> m_OrientationsDrawables;

    osg::Geode* drawCam();
    osg::Vec4 m_Color{0,1,0,1};
    float m_Scale{1};
};

