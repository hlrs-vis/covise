/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
/****************************************************************************\
 **                                                            (C)2020 HLRS  **
 **                                                                          **
 ** Description: Camera position and orientation optimization                **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple	                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April 2020  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <vector>
#include <memory>


#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/Owner.h>

#include "Zone.h"
struct SensorResolution
{
    float rotz,rotx,roty;
};

template<typename T>
class VisibilityMatrix
{
public:
    VisibilityMatrix(std::vector<T> visMat):m_VisibilityMatrix(visMat){};
    void updateVisibilityMatrix(std::vector<T>&& visMat);

private:
    std::vector<T> m_VisibilityMatrix;
};

class Sensor;
class Orientation
{
public:
    Orientation(osg::Matrix matrix,std::vector<float> visMat):m_Matrix(matrix),m_VisibilityMatrix(visMat){};
    Orientation(osg::Matrix matrix);

    bool operator>>(const Sensor& other)const;

    osg::Matrix const& getMatrix()const{return m_Matrix;}
    VisibilityMatrix<float> const& getVisibilityMatrix()const{return m_VisibilityMatrix;}

    void setMatrix(osg::Matrix m){m_Matrix = m;}

private:
    osg::Matrix m_Matrix;
    VisibilityMatrix<float> m_VisibilityMatrix;
};



class Sensor
{
public:
    explicit Sensor(osg::Matrix matrix):m_Orientation(matrix){};
    virtual ~Sensor();
    virtual void calcVisibilityMatrix() = 0;
    virtual void draw() = 0;

    bool calcIntersection(osg::Vec3& point);

    void setPosition(osg::Matrix matrix){m_Orientation.setMatrix(matrix);}
    const VisibilityMatrix<float>& getVisibilityMatrix() {return m_Orientation.getVisibilityMatrix();}
    const osg::Matrix getPosition()const{return m_Orientation.getMatrix();}
        
private:
    Orientation m_Orientation;
    std::vector<Orientation> m_PossibleOrientations;

    SensorResolution m_SensorProps;
};


struct CameraProps
{   
    float m_FoV = 60;
    float m_DepthView = 40;
 
    float m_ImgWidth = 2*m_DepthView*std::tan(m_FoV/2*osg::PI/180);
    float m_ImgHeight = m_ImageWidthPixel/m_ImageHeightPixel;

    void updateFoV(float fov);
    void updateDepthView(float dof);

private:
    // Full HD Camera
    float m_ImageHeightPixel = 1080;
    float m_ImageWidthPixel = 1920;

};

class Camera: public Sensor
{
public:
    explicit Camera(osg::Matrix matrix):Sensor(matrix){};
    void calcVisibilityMatrix() override;
  
private:
    CameraProps m_CameraProps;
    std::vector<Orientation> m_Orientations;

};
class SafetyZone
{

};

class Camera;
class SafetyZone;

typedef std::unique_ptr<Camera> upCamera;
typedef std::unique_ptr<SafetyZone> upSafetyZone;

//Singleton Class
class Data
{
public:
    Data(const Data& other) = delete;
    Data operator=(const Data& other) = delete;
    
    static Data& GetInstance()
    {
        static Data instance;
        return instance;
    }

    static const std::vector<upCamera>& GetCameras(){return GetInstance().m_Cameras;}
    static const std::vector<upSafetyZone>& GetSafetyZones(){return GetInstance().m_SafetyZones;}

    static void AddCamera(upCamera camera)
    {
        GetInstance().m_Cameras.push_back(std::move(camera));
    }
    static void AddSafetyZone(upSafetyZone safetyZone)
    {
        GetInstance().m_SafetyZones.push_back(std::move(safetyZone));
    }

private:
    Data(){}

    std::vector<upCamera> m_Cameras;
    std::vector<upSafetyZone> m_SafetyZones;

    std::vector<std::future<void>> m_Futures; 
};

class SensorPlacementPlugin :public opencover::coVRPlugin, public opencover::ui::Owner
{
    public:
    SensorPlacementPlugin();
    ~SensorPlacementPlugin();
    bool init() override;
    void preFrame() override;

    private:
    Zone* zone;
};