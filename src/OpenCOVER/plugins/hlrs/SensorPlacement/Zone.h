#pragma once

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransInteractor.h>

#include <osg/ShapeDrawable>
#include <osg/Geometry>

using namespace opencover;


class GridPoint;
class SensorPosition;

struct ZoneProperties
{

};
/*   * Vertices of the Zone 
     *    4*    *7
     *
     * 5*----*6
     *  |0*  |  *3
     *  |    |
     * 1*----*2
     * 
     * Coordinate System is centered at 2
     * width = 7-6
     * length = 5-6
*/

class Zone
{
public:
    Zone(osg::Matrix matrix);
    virtual ~Zone(){std::cout<<"Zone Destructor\n";};
    virtual void createGrid() = 0;

    bool preFrame();
    
    void setPosition(osg::Matrix matrix);
    void setDistance(float distance);

    osg::ref_ptr<osg::MatrixTransform> getZone(){return m_LocalDCS;}
    int getNumberOfPoints()const{return m_GridPoints.size();}

    std::vector<osg::Vec3> getWorldPositionOfPoints();
private:
    float m_Distance{2};
    float m_Length{10};
    float m_Width{5};
    float m_Height{3};
    osg::Vec4 m_Color{1,0,0,1};
    
    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Geometry> m_Geom;
    osg::ref_ptr<osg::Geode> m_Geode;

    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    std::unique_ptr<coVR3DTransRotInteractor> m_Interactor;
    std::unique_ptr<coVR3DTransInteractor> m_SizeInteractor;
    std::unique_ptr<coVR3DTransInteractor> m_DistanceInteractor;

    std::vector<GridPoint> m_GridPoints;

    osg::Geode* draw();

    void createGridPoints();
    void deleteGridPoints();
    void create3DGrid(const osg::Vec3& startPoint, const osg::Vec3& signconst, float widthLimit, const float lengthLimit, const float heightLimit);
    void updateGeometry(osg::Vec3& vec);
    void deletePoints();

};

struct SafetyZoneProperties
{
    enum
    {
        PRIO1,
        PRIO2
    };

};

class SafetyZone : public Zone
{

public:
    SafetyZone(osg::Matrix matrix):Zone(matrix){std::cout<<"Safety Zone created\n";}
    ~SafetyZone(){std::cout<<"SafetyZone Destructor\n";};
    void createGrid()override{};

private:
    osg::Vec4 m_ColorVisible;


};

struct SensorZoneProperties
{

};

class SensorZone : public Zone
{
public:
    SensorZone(osg::Matrix matrix):Zone(matrix){std::cout<<"Sensor Zone created\n";}
    ~SensorZone(){std::cout<<"SensorZone Destructor\n";};
    void createGrid()override{};


private:
    std::unique_ptr<SensorPosition> m_PSensor;  

};


class GridPoint
{
public:
    GridPoint(osg::Vec3 position,osg::Vec4& color);
    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    osg::ref_ptr<osg::MatrixTransform> getPoint()const{return m_LocalDCS.get();} //muss man hier ref_ptr übergeben?
    osg::Vec3 getPosition()const{return m_LocalDCS->getMatrix().getTrans();}

    void setColor(const osg::Vec4& color);
    void setOriginalColor();
private:
    osg::Vec4 m_Color;
    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Sphere> m_Sphere;
    osg::ref_ptr<osg::ShapeDrawable> m_SphereDrawable;
};
