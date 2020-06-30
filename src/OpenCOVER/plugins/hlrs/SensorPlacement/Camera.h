#pragma once

#include "Sensor.h"

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coVRPluginSupport.h>

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

    //void updateFoV(float fov);
    //void updateDepthView(float dof);

    int getImageHeightPixel()const{return m_ImageHeightPixel;}
    int getImageWidthPixel()const{return m_ImageWidthPixel;}

};

class Camera : public SensorWithMultipleOrientations
{
public:
   
    Camera(osg::Matrix matrix, bool visible);
    ~Camera()override{};
    
    bool preFrame() override;
    osg::Geode* draw() override;
    VisibilityMatrix<float> calcVisibilityMatrix(coCoord& euler) override;
    void VisualizationVisible(bool status)const override;

    const CameraProps& getCameraProps()const {return m_CameraProps;}

protected:
    double calcRangeDistortionFactor(const osg::Vec3& point)const override;
    double calcWidthDistortionFactor(const osg::Vec3& point)const override;
    double calcHeightDistortionFactor(const osg::Vec3& point)const override;

    osg::Geode* drawOrientation() override;
    void showOriginalSensorSize() override;
    void showIconSensorSize() override;

private:
    CameraProps m_CameraProps;

    osg::Vec4 m_Color{0,1,0,1};
    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Vec4Array> m_Colors;
    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Geometry> m_Geometry;

    osg::ref_ptr<osg::Vec3Array> m_VertsOrientation;
    osg::ref_ptr<osg::Geode> m_GeodeOrientation;
    osg::ref_ptr<osg::Geometry> m_GeometryOrientation;
};

