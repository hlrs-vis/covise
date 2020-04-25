#pragma once

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransInteractor.h>

using namespace opencover;

class GridPoint;

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
    //virtual ~Zone();  --> make Destructor virtual!

    void preFrame();
    
    void setPosition(osg::Matrix matrix);
    void setDistance(float distance);

    int getNbrControlPoints()const;

private:

    float m_Distance;
    float m_Length{1};
    float m_Width{1};
    float m_Height{0.1};
    osg::Vec4 m_Color;
    
    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Geometry> m_Geom;
    osg::ref_ptr<osg::Geode> m_Geode;

    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    std::unique_ptr<coVR3DTransRotInteractor> m_Interactor;
    std::unique_ptr<coVR3DTransInteractor> m_SizeInteractor;

    

    std::vector<GridPoint> m_GridPoints;

    osg::Geode* draw();
   // void setStateSet(osg::StateSet *stateSet);

    void createGridPoints();
    void deleteGridPoints();
    void updateGeometry(osg::Vec3& vec);
    void createPoints();
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
/*
class SafetyZone : public Zone
{

public:

private:
    osg::Vec4 m_ColorVisible;


};

struct SensorZoneProperties
{

};

class SensorZone : public Zone
{
public:

private:
};
*/

class GridPoint
{
public:
    GridPoint(osg::Vec3 position);
};