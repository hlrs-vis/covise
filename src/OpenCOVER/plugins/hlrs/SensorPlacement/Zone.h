#pragma once

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransInteractor.h>

#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osgText/Text>

#include "Sensor.h"
using namespace opencover;


float findLargestVectorComponent(const osg::Vec3& vec);
class Zone;
class SensorZone;
class SafetyZone;
class GridPoint;


/*   * Vertices of a RectangleZone 
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
class ZoneShape
{
public:
    virtual bool preFrame(){return true;}
    virtual void setPosition(osg::Matrix matrix);
    virtual void hide(){};
    virtual void show(){};
    float getDistance()const {return m_Distance;}
    osg::Matrix getMatrix()const{return m_LocalDCS->getMatrix();}
    const osg::ref_ptr<osg::MatrixTransform>& getMatrixTransform()const{return m_LocalDCS;}
    const std::vector<GridPoint>& getGridPoints()const{return m_GridPoints;}
    std::vector<GridPoint>& getGridPoints(){return m_GridPoints;}


    ZoneShape(osg::Matrix matrix, osg::Vec4 color, Zone* zone);
    virtual ~ZoneShape(){};


protected:
    Zone* m_ZonePointer;
    float m_Distance;
    osg::Vec4 m_Color;
    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    osg::ref_ptr<osg::Geode> m_Geode;
    std::vector<GridPoint> m_GridPoints;

    //virtual osg::Geode* draw();
    virtual void drawGrid() = 0;
    void deleteGridPoints();
    void addPointToVec(osg::Vec3 point);
    virtual float calcGridPointDiameter() = 0;

};

class ZoneSphere : public ZoneShape
{
public:
    ZoneSphere(osg::Matrix matrix,float radius, osg::Vec4 color, Zone* zone);
    virtual ~ZoneSphere(){};

private:
    float m_Radius;

    //osg::Geode* draw(){return nullptr;} override;
    void drawGrid() override;
    void createCircle();
    float calcGridPointDiameter() override;
    std::vector<osg::Vec3> circleVerts(osg::Vec3 axis, float radius, int approx, float height);
};

class ZoneRectangle : public ZoneShape
{
public:
    ZoneRectangle(osg::Matrix matrix,float length, float width, float height,osg::Vec4 color,Zone* zone);
    virtual ~ZoneRectangle(){};
    bool preFrame() override;
    void setPosition(osg::Matrix matrix)override;

    void updateGeometry(osg::Vec3& vec);
    void show() override;
    void hide() override;

    //osg::Vec3 getVertex(int vertPos)const;
private:
    float m_Length;
    float m_Width;
    float m_Height;

    osg::ref_ptr<osg::Vec3Array> m_Verts;
    osg::ref_ptr<osg::Vec4Array> m_Colors;
    osg::ref_ptr<osg::Geometry> m_Geom;

    std::unique_ptr<coVR3DTransRotInteractor> m_Interactor; // located at 2
    std::unique_ptr<coVR3DTransInteractor> m_SizeInteractor; // located at 0
    std::unique_ptr<coVR3DTransInteractor> m_DistanceInteractor; // always located along longest side

    osg::Geode* draw();// override;
    void drawGrid() override;


    osg::Vec3 findLongestSide()const;
    float calcGridPointDiameter() override;

    void calcPositionOfDistanceInteractor(osg::Vec3& startPosInteractor);
    void restrictDistanceInteractor(osg::Vec3& startPosInteractor);

    osg::Vec3 calcSign()const;
    osg::Vec3 defineStartPointForInnerGrid()const;
    osg::Vec3 defineLimitsOfInnerGridPoints()const;
    void createInner3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign, const osg::Vec3& limit);
    void createOuter3DGrid(const osg::Vec3& sign );
    float calculateGridPointDistance() const;

};


class Zone
{
public:

    Zone( osg::Matrix matrix,osg::Vec4 color, float length , float width , float height );
    Zone(osg::Matrix matrix, osg::Vec4 color, float radius);
    virtual ~Zone() = 0;

    virtual bool preFrame(){return m_Shape->preFrame();};
    
    void setPosition(osg::Matrix matrix){m_Shape->setPosition(matrix);}
    void hide(){m_Shape->hide();}
    void show(){m_Shape->show();}

    osg::ref_ptr<osg::Group> getZone(){return m_Group;}
    int getNumberOfPoints()const{return m_Shape->getGridPoints().size();}

    std::vector<osg::Vec3> getWorldPositionOfPoints();
    
    void setOriginalColor();

    virtual void setCurrentNbrOfSensors(int sensors){}; // shuld only be available in Safety Zone -> but couldn't call

protected:
    osg::Vec4 m_Color;

    //virtual void createGrid() = 0;

    // osg::Vec3 calcSign()const;
    // osg::Vec3 defineStartPointForInnerGrid()const;
    // osg::Vec3 defineLimitsOfInnerGridPoints()const;
    // void createInner3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign, const osg::Vec3& limit);
    // void createOuter3DGrid(const osg::Vec3& sign );
    
    std::unique_ptr<ZoneShape> m_Shape;   

    osg::ref_ptr<osg::Group> m_Group;
    //osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

private: 
   
    //void drawGrid(); // calls virtual createGrid() function

    // void calcPositionOfDistanceInteractor(osg::Vec3& startPosInteractor);
    // void restrictDistanceInteractor(osg::Vec3& startPosInteractor);
    //osg::Vec3 findLongestSide()const;
   
    //void addPointToVec(osg::Vec3 point);
    //void deleteGridPoints();
    
    //void updateGeometry(osg::Vec3& vec);

    //float calculateGridPointDistance() const;

};


class SafetyZone : public Zone
{

public:
    enum struct Priority{PRIO1 = 2, PRIO2 = 1};

    SafetyZone(osg::Matrix matrix, Priority priority, float length, float width, float height);
    SafetyZone(osg::Matrix matrix, Priority priority, float radius);
    ~SafetyZone(){std::cout<<"SafetyZone Destructor\n";};
    void setCurrentNbrOfSensors(int sensors) override;                            

    Priority getPriority()const {return m_Priority;}
    void highlitePoints(VisibilityMatrix<float>& visiblePoints);
    void highlitePoints(VisibilityMatrix<float>& visiblePoints, osg::Vec4& colorVisible, osg::Vec4& colorNotVisible);
    void setPreviousZoneColor();

private:
    Priority m_Priority;
    int m_CurrentNbrOfSensors{0}; // Nbr of Sensors which currently see this zone

    osg::ref_ptr<osgText::Text> m_Text;

    osg::Vec4 calcColor( Priority prio) const;
    //void createGrid() override;    

};

enum class SensorType;
class SensorZone : public Zone
{

public:
    SensorZone(SensorType type, osg::Matrix matrix,float length, float width, float height);
    SensorZone(SensorType type, osg::Matrix matrix, float radius);

    ~SensorZone(){std::cout<<"SensorZone Destructor\n";};
    virtual bool preFrame()override;

    //void createGrid() override;
    void createAllSensors();
    SensorPosition* getSpecificSensor(int position) const {return m_Sensors.at(position).get();}
    std::vector<std::unique_ptr<SensorPosition>>& getSensors(){return m_Sensors;}
    const std::vector<std::unique_ptr<SensorPosition>>& getSensors()const{return m_Sensors;}

    int getTargetNumberOfSensors(){return m_NbrOfSensors;}
    int getActualNumberOfSensors(){return m_Sensors.size();}
    //void crgeateSpecificNbrOfSensors(const std::vector<osg::Matrix>& sensorMatrixes); //This function creates the sensors defined in function input
    void createSpecificNbrOfSensors(); //This function creates as many sensors as in m_NbrOfSensors defined

    void createSensor(const osg::Matrix& matrix);
    void removeAllSensors();
    void updateFoV(float fov);
    void updateDoF(float dof);
 
private:
    int m_NbrOfSensors{1};
    SensorType m_SensorType;
    std::vector<std::unique_ptr<SensorPosition>> m_Sensors; 
    osg::ref_ptr<osg::Group> m_SensorGroup;
 

    void addSensor(osg::Matrix matrix, bool visible);
    osg::Vec3 getFreeSensorPosition()const;

    
};


class GridPoint
{
public:
    GridPoint(osg::Vec3 position,osg::Vec4& color, float radius);
    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    osg::ref_ptr<osg::MatrixTransform> getPoint()const{return m_LocalDCS.get();} //muss man hier ref_ptr übergeben?
    osg::Vec3 getPosition()const{return m_LocalDCS->getMatrix().getTrans();}

    void highlite(const osg::Vec4& color);
    void setColor(const osg::Vec4& color);
    void setOriginalColor();        //sets the color to m_Color
    void setPreviousColor();

private:
    osg::Vec4 m_Color;
    osg::Vec4 m_PreviousColor; //color: is this zone observed by enough cameras or not

    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Sphere> m_Sphere;
    osg::ref_ptr<osg::ShapeDrawable> m_SphereDrawable;
};
