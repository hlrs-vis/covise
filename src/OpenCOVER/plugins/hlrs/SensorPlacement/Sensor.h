#pragma once

#include<vector>
#include<memory>

#include<osg/Matrix>
#include<osg/Geode>
#include<osg/Geometry>
#include<osg/MatrixTransform>

#include <PluginUtil/coVR3DTransRotInteractor.h>

struct VisbilityMatrix;
void checkForObstacles(osg::Vec3& sensorPos);
void checkVisibility(const osg::Matrix& sensorMatrix, VisbilityMatrix& visMat);

struct VisbilityMatrix
{
public:
   // VisbilityMatrix(std::vector<float>&& visMat):m_VisibilityMatrix(visMat){};
    std::vector<float> m_VisibilityMatrix;
};
class Orientation
{
    public:
       // Orientation(osg::Matrix matrix, VisbilityMatrix visMat)
       //     :m_Matrix(matrix),m_VisibilityMatrix(visMat){};
    Orientation(osg::Matrix matrix):m_Matrix(matrix){};

    bool operator >> (const Orientation& other) const;

    private:
    osg::Matrix m_Matrix;
    VisbilityMatrix m_VisibilityMatrix;
};

class SensorPosition
{

    public:
    SensorPosition(osg::Matrix matrix);
    virtual ~SensorPosition(){}; //do I now need to implement move ....
    virtual void calcVisibilityMatrix() = 0;
    virtual bool preFrame() = 0;
    virtual osg::ref_ptr<osg::Group> getSensor() = 0;
    
    void checkForObstacles();

    private:
    Orientation m_Orientation; //each Sensor has at least one Orientation / Position 
   // SensorGraphics m_graphicalRepresentation;


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
    private:
        SensorProps m_SensorProps;
        std::vector<Orientation> m_Orientations;
       // SensorGraphics m_graphicalRepresentationOfOrientations;
    public:

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


};

class Camera : public SensorWithMultipleOrientations
{

public:
    Camera(osg::Matrix matrix);
    ~Camera()override{};
    void calcVisibilityMatrix() override{}
    bool preFrame() override;
    osg::ref_ptr<osg::Group> getSensor() override{return m_CameraMatrix;}



private:
    CameraProps m_CameraProps;
    std::unique_ptr<opencover::coVR3DTransRotInteractor> m_Interactor;

    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Vec4Array> m_Colors;
    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Geometry> m_Geometry;
    osg::ref_ptr<osg::MatrixTransform> m_CameraMatrix;


    osg::Geode* drawCam();
    osg::Vec4 m_Color{0,1,0,1};
    float m_Scale{1};
};
