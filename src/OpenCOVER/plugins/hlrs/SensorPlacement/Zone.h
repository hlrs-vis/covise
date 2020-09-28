#pragma once

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransInteractor.h>

#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osgText/Text>

#include "Sensor.h"
using namespace opencover;


class GridPoint;

struct ZoneProperties
{

};
/*   * Vertices of a Zone 
     *    4*    *7
     *
     * 5*----*6
     *  |0*  |  *3
     *  |    |
     * 1*----*2
     * 
     * Coordinate System is centered at 2
     * Size interactor located at 0
     * width = 7-6 (y-direction)
     * length = 5-6 (x-direction)
     * height = 6-2 (z-direction)
*/

class Zone
{
public:
    Zone(osg::Matrix matrix,osg::Vec4 color, float length = 10.0f, float width = 5.0f, float height = 3.0f);
    virtual ~Zone(){std::cout<<"Zone Destructor\n";};

    virtual bool preFrame();
    
    void setPosition(osg::Matrix matrix);
    void setDistance(float distance);

    osg::ref_ptr<osg::Group> getZone(){return m_Group;}
    int getNumberOfPoints()const{return m_GridPoints.size();}

    std::vector<osg::Vec3> getWorldPositionOfPoints();
    void highlitePoints(VisibilityMatrix<float>& visiblePoints);
    void setOriginalColor();

    virtual void setCurrentNbrOfSensors(int sensors){}; // shuld only be available in Safety Zone -> but couldn't call

protected:
    osg::Vec4 m_Color;

    virtual void createGrid() = 0;

    osg::Vec3 calcSign()const;
    osg::Vec3 defineStartPointForInnerGrid()const;
    osg::Vec3 defineLimitsOfInnerGridPoints()const;
    void createInner3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign, const osg::Vec3& limit);
    void createOuter3DGrid(const osg::Vec3& sign );

    osg::ref_ptr<osg::Group> m_Group;
    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;
    std::vector<GridPoint> m_GridPoints;

    std::unique_ptr<coVR3DTransRotInteractor> m_Interactor; // located at 2
    std::unique_ptr<coVR3DTransInteractor> m_SizeInteractor; // located at 0
    std::unique_ptr<coVR3DTransInteractor> m_DistanceInteractor; // always located along longest side


private:
    float m_Distance;
    float m_Length;
    float m_Width;
    float m_Height;
    
    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Vec4Array> m_Colors;
    osg::ref_ptr<osg::Geometry> m_Geom;
    osg::ref_ptr<osg::Geode> m_Geode;

    osg::Geode* draw();

    void drawGrid(); // calls virtual createGrid() function

    void calcPositionOfDistanceInteractor(osg::Vec3& startPosInteractor);
    void restrictDistanceInteractor(osg::Vec3& startPosInteractor);
    osg::Vec3 findLongestSide()const;
    float findLargestVectorComponent(osg::Vec3 vec) const;
    float calculateGridPointDistance() const;
    void addPointToVec(osg::Vec3 point);
    void deleteGridPoints();
    
    void updateGeometry(osg::Vec3& vec);

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
    SafetyZone(osg::Matrix matrix,float length = 10.0f, float width = 5.0f, float height = 3.0f);
    ~SafetyZone(){std::cout<<"SafetyZone Destructor\n";};
    void setCurrentNbrOfSensors(int sensors) override;                            

    
private:
    void createGrid() override;    
    osg::Vec4 m_ColorVisible;
    int m_CurrentNbrOfSensors; // Nbr of Sensors which currently see this zone

    osg::ref_ptr<osgText::Text> m_Text;

};

struct SensorZoneProperties
{

};

enum class SensorType;
class SensorZone : public Zone
{

public:
    SensorZone(SensorType type,osg::Matrix matrix,float length = 10.0f, float width = 5.0f, float height = 3.0f);
    ~SensorZone(){std::cout<<"SensorZone Destructor\n";};
    bool preFrame() override;
    void createGrid() override;
    void createAllSensors();
    SensorPosition* getSpecificSensor(int position) const {return m_Sensors.at(position).get();}
    int getNumberOfSensors(){return m_NbrOfSensors;}

private:
    int m_NbrOfSensors{2};
    SensorType m_SensorType;
    std::vector<std::unique_ptr<SensorPosition>> m_Sensors; 
    osg::ref_ptr<osg::Group> m_SensorGroup;
 
    void createSpecificNbrOfSensors();
    void addSensor(osg::Matrix matrix, bool visible);
    osg::Vec3 getFreeSensorPosition()const;
    void removeAllSensors();
};


class GridPoint
{
public:
    GridPoint(osg::Vec3 position,osg::Vec4& color, float radius);
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
