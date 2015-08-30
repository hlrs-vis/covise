/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef RSYSTEMELEMENTROAD_HPP
#define RSYSTEMELEMENTROAD_HPP

#include "rsystemelement.hpp"

#include <QMap>
#include <QPointF>
#include <QTransform>
#include <QVector2D>
#include <QStringList>

class RoadLink;

class RoadSection;
class TypeSection;
class SurfaceSection;
class TrackComponent;
class ElevationSection;
class SuperelevationSection;
class CrossfallSection;
class LaneSection;
class Object;
class Crosswalk;
class Signal;
class Sensor;
class Surface;
class Bridge;

class RSystemElementRoad : public RSystemElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum DRoadSectionType
    {
        DRS_TypeSection = 0x1,
        DRS_TrackSection = 0x2,
        DRS_ElevationSection = 0x4,
        DRS_SuperelevationSection = 0x10,
        DRS_CrossfallSection = 0x20,
        DRS_LaneSection = 0x40,
        DRS_SignalSection = 0x50,
        DRS_ObjectSection = 0x60,
        DRS_BridgeSection = 0x70
    };

    enum DRoadObjectType
    {
        DRO_Crosswalk = 0x1,
        DRO_Signal = 0x2,
        DRO_Sensor = 0x4,
        DRO_Surface = 0x10,
        DRO_Object = 0x20
    };

    enum DContactPoint
    {
        DRC_Start = 0x1,
        DRC_End = 0x2,
        DRC_Unknown = 0x4,
    };

    enum RoadChange
    {
        CRD_TypeSectionChange = 0x1,
        CRD_TrackSectionChange = 0x2,
        CRD_ElevationSectionChange = 0x4,
        CRD_SuperelevationSectionChange = 0x10,
        CRD_CrossfallSectionChange = 0x20,
        CRD_LaneSectionChange = 0x40,
        CRD_JunctionChange = 0x100,
        CRD_LengthChange = 0x200,
        CRD_PredecessorChange = 0x400,
        CRD_SuccessorChange = 0x800,
        CRD_ShapeChange = 0x1000,
        CRD_ObjectChange = 0x2000,
        CRD_CrosswalkChange = 0x4000,
        CRD_SignalChange = 0x8000,
        CRD_SensorChange = 0x10000,
        CRD_SurfaceChange = 0x20000,
        CRD_BridgeChange = 0x40000
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RSystemElementRoad(const QString &name, const QString &id, const QString &junction);
    virtual ~RSystemElementRoad();

    // road //
    //
    QString getJunction() const
    {
        return junction_;
    }
    void setJunction(const QString &junctionId);

    double getLength() const
    {
        return cachedLength_;
    }

    bool moveRoadSection(RoadSection *section, double newS, RSystemElementRoad::DRoadSectionType sectionType);
    RoadSection *getRoadSectionBefore(double s, RSystemElementRoad::DRoadSectionType sectionType) const;

    // road:link //
    //
    RoadLink *getPredecessor() const
    {
        return predecessor_;
    }
    RoadLink *getSuccessor() const
    {
        return successor_;
    }
    bool delPredecessor();
    bool delSuccessor();

    void setPredecessor(RoadLink *roadLink);
    void setSuccessor(RoadLink *roadLink);

    // road:type //
    //
    void addTypeSection(TypeSection *section);
    bool delTypeSection(TypeSection *section);
    bool moveTypeSection(double oldS, double newS);
    void setTypeSections(QMap<double, TypeSection *> newSections);

    TypeSection *getTypeSection(double s) const;
    TypeSection *getTypeSectionBefore(double s) const;
    double getTypeSectionEnd(double s) const;
    QMap<double, TypeSection *> getTypeSections() const
    {
        return typeSections_;
    }

    // road:surface //
    //
    void addSurfaceSection(SurfaceSection *section);
    SurfaceSection *getSurfaceSections() const
    {
        return surfaceSection_;
    }

    // road:planView (track) //
    //
    void addTrackComponent(TrackComponent *track);
    bool delTrackComponent(TrackComponent *track);
    void setTrackComponents(QMap<double, TrackComponent *> newSections);

    TrackComponent *getTrackComponent(double s) const;
    TrackComponent *getTrackComponentBefore(double s) const;
    TrackComponent *getTrackComponentNext(double s) const;
    QMap<double, TrackComponent *> getTrackSections() const
    {
        return trackSections_;
    }

    void rebuildTrackComponentList();

    QPointF getGlobalPoint(double s, double d = 0.0) const;
    double getGlobalHeading(double s) const;
    QVector2D getGlobalTangent(double s) const;
    QVector2D getGlobalNormal(double s) const;
    QTransform getGlobalTransform(double s, double d = 0.0) const;

    double getSFromGlobalPoint(const QPointF &globalPos, double sInit = -1.0);
    double getSFromGlobalPoint(const QPointF &globalPos, double sStart, double sEnd);

    // road:elevationProfile:elevation //
    //
    void addElevationSection(ElevationSection *section);
    bool delElevationSection(ElevationSection *section);
    bool moveElevationSection(double oldS, double newS);
    void setElevationSections(QMap<double, ElevationSection *> newSections);

    ElevationSection *getElevationSection(double s) const;
    ElevationSection *getElevationSectionBefore(double s) const;
    ElevationSection *getElevationSectionNext(double s) const;
    double getElevationSectionEnd(double s) const;
    QMap<double, ElevationSection *> getElevationSections() const
    {
        return elevationSections_;
    }

    // road:lateralProfile:superelevation //
    //
    void addSuperelevationSection(SuperelevationSection *section);
    bool delSuperelevationSection(SuperelevationSection *section);
    bool moveSuperelevationSection(double oldS, double newS);
    void setSuperelevationSections(QMap<double, SuperelevationSection *> newSections);

    SuperelevationSection *getSuperelevationSection(double s) const;
    SuperelevationSection *getSuperelevationSectionBefore(double s) const;
    double getSuperelevationSectionEnd(double s) const;
    QMap<double, SuperelevationSection *> getSuperelevationSections() const
    {
        return superelevationSections_;
    }

    // road:lateralProfile:crossfall //
    //
    void addCrossfallSection(CrossfallSection *section);
    bool delCrossfallSection(CrossfallSection *section);
    bool moveCrossfallSection(double oldS, double newS);
    void setCrossfallSections(QMap<double, CrossfallSection *> newSections);

    CrossfallSection *getCrossfallSection(double s) const;
    CrossfallSection *getCrossfallSectionBefore(double s) const;
    double getCrossfallSectionEnd(double s) const;
    QMap<double, CrossfallSection *> getCrossfallSections() const
    {
        return crossfallSections_;
    }

    // road:laneSection //
    //
    void addLaneSection(LaneSection *laneSection);
    bool delLaneSection(LaneSection *laneSection);
    bool moveLaneSection(double oldS, double newS);
    void setLaneSections(QMap<double, LaneSection *> newSections);

    LaneSection *getLaneSection(double s) const;
    LaneSection *getLaneSectionBefore(double s) const;
    LaneSection *getLaneSectionNext(double s) const;
    double getLaneSectionEnd(double s) const;
    QMap<double, LaneSection *> getLaneSections() const
    {
        return laneSections_;
    }

    // road:laneSection:lane:width //
    //
    double getMaxWidth(double s) const;
    double getMinWidth(double s) const;

    // road:objects:crosswalk //
    //
    void addCrosswalk(Crosswalk *crosswalk);
    QMap<double, Crosswalk *> getCrosswalks() const
    {
        return crosswalks_;
    }
    bool delCrosswalk(Crosswalk *crosswalk);

    // road:objects:object //
    //
    void addObject(Object *object);
    QMap<double, Object *> getObjects() const
    {
        return objects_;
    }
    bool delObject(Object *object);
    bool moveObject(RoadSection *section, double newS);

    // road:objects:bridge //
    //
    void addBridge(Bridge *bridge);
    QMap<double, Bridge *> getBridges() const
    {
        return bridges_;
    }
    bool delBridge(Bridge *bridge);
    bool moveBridge(RoadSection *section, double newS);

    // road:objects:signal //
    //
    void addSignal(Signal *signal);
    QMultiMap<double, Signal *> getSignals() const
    {
        return signals_;
    }
    bool delSignal(Signal *signal);
    bool moveSignal(RoadSection *section, double newS);
    int getValidLane(double s, double t);
    Signal * getSignal(const QString &id);

    // road:objects:sensor //
    //
    void addSensor(Sensor *sensor);
    QMap<double, Sensor *> getSensors() const
    {
        return sensors_;
    }
    bool delSensor(Sensor *sensor);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRoadChanges() const
    {
        return roadChanges_;
    }
    void addRoadChanges(int changes);

    // Prototype Pattern //
    //
    void superposePrototype(const RSystemElementRoad *prototypeRoad);

    RSystemElementRoad *getClone() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

    virtual void acceptForChildNodes(Visitor *visitor);

    virtual void acceptForRoadLinks(Visitor *visitor);
    virtual void acceptForTypeSections(Visitor *visitor);
    virtual void acceptForSurfaceSections(Visitor *visitor);
    virtual void acceptForTracks(Visitor *visitor);
    virtual void acceptForElevationSections(Visitor *visitor);
    virtual void acceptForSuperelevationSections(Visitor *visitor);
    virtual void acceptForCrossfallSections(Visitor *visitor);
    virtual void acceptForLaneSections(Visitor *visitor);
    virtual void acceptForObjects(Visitor *visitor);
    virtual void acceptForBridges(Visitor *visitor);
    virtual void acceptForCrosswalks(Visitor *visitor);
    virtual void acceptForSignals(Visitor *visitor);
    virtual void acceptForSensors(Visitor *visitor);
    
    double updateLength();

private:
    RSystemElementRoad(); /* not allowed */
    RSystemElementRoad(const RSystemElementRoad &); /* not allowed */
    RSystemElementRoad &operator=(const RSystemElementRoad &); /* not allowed */


    // IDs //
    //
    const QString getUniqueId(const QString &suggestion, RSystemElement::DRoadSystemElementType elementType);

    bool delTypeSection(double s);
    bool delTrackComponent(double s);
    bool delElevationSection(double s);
    bool delSuperelevationSection(double s);
    bool delCrossfallSection(double s);
    bool delLaneSection(double s);
    bool delObject(double s);
    bool delBridge(double s);
    bool delCrosswalk(double s);
    bool delSignal(double s);
    bool delSensor(double s);
    bool delSurface(double s);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change //
    //
    int roadChanges_;

    // road //
    //
    QString junction_; // ID of the junction (if road is a path), otherwise -1
    double cachedLength_; // total length of the road (xy-plane)

    // link //
    //
    RoadLink *predecessor_; // owned
    RoadLink *successor_; // owned

    // Sections //
    //
    // type //
    QMap<double, TypeSection *> typeSections_; // owned

    // surface //
    SurfaceSection *surfaceSection_; // owned

    // planView //
    QMap<double, TrackComponent *> trackSections_; // owned

    // elevation //
    QMap<double, ElevationSection *> elevationSections_; // owned

    // superelevation //
    QMap<double, SuperelevationSection *> superelevationSections_; // owned

    // crossfall //
    QMap<double, CrossfallSection *> crossfallSections_; // owned

    // lanes //
    QMap<double, LaneSection *> laneSections_; // owned

    // objects //
    QMap<double, Crosswalk *> crosswalks_; // owned

    QMultiMap<double, Object *> objects_; // owned
    QStringList objectIds_;
    int objectIdCount_;

    QMultiMap<double, Bridge *> bridges_; // owned

    QMultiMap<double, Signal *> signals_; // owned
    QStringList signalIds_;
    int signalIdCount_;

    QMap<double, Sensor *> sensors_; // owned
    QMap<double, Surface *> surfaces_; // owned
};

#endif // RSYSTEMELEMENTROAD_HPP
