/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.05.2010
**
**************************************************************************/

#ifndef PROTOTYPE_HPP
#define PROTOTYPE_HPP

#include <QObject>

#include <QString>
#include <QIcon>
#include <QMultiMap>

#include <src/data/roadsystem/rsystemelementroad.hpp>
#include <src/data/roadsystem/roadsystem.hpp>

class TrackComponent;

template <class T_DERIVED>
class PrototypeContainer
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PrototypeContainer(const QString &name, const QIcon &icon, T_DERIVED prototype,QString &system,QString &tn,QString &ln)
        : prototypeName_(name)
        , prototypeIcon_(icon)
        , prototype_(prototype)
        , systemName(system)
        , typeName(tn)
        , laneNumbers(ln)
    {
        /* does nothing */
    }
    explicit PrototypeContainer(const QString &name, const QIcon &icon, T_DERIVED prototype)
        : prototypeName_(name)
        , prototypeIcon_(icon)
        , prototype_(prototype)
    {
        /* does nothing */
    }

    virtual ~PrototypeContainer()
    {
        delete prototype_;
    }

    QString getPrototypeName() const
    {
        return prototypeName_;
    }
    QIcon getPrototypeIcon() const
    {
        return prototypeIcon_;
    }
    T_DERIVED getPrototype() const
    {
        return prototype_;
    }
    void setSystemName(QString &s)
    {
        systemName = s;
    }
    void setTypeName(QString &s)
    {
        typeName = s;
    }
    void setLaneNumbers(QString &s)
    {
        laneNumbers = s;
    }
    QString &getLaneNumbers()
    {
        return laneNumbers;
    }
    QString &getTypeName()
    {
        return typeName;
    }
    QString &getSystemName()
    {
        return systemName;
    }

protected:
private:
    PrototypeContainer(); /* not allowed */
    PrototypeContainer(const PrototypeContainer &); /* not allowed */
    PrototypeContainer &operator=(const PrototypeContainer &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    QString prototypeName_;
    QIcon prototypeIcon_;
    T_DERIVED prototype_;
    
    QString systemName;
    QString typeName;
    QString laneNumbers;
};

class PrototypeManager : public QObject
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    enum PrototypeType
    {
        PTP_RoadPrototype = 0x1,
        PTP_RoadTypePrototype = 0x2,
        PTP_TrackPrototype = 0x4,
        PTP_ElevationPrototype = 0x8,
        PTP_SuperelevationPrototype = 0x10,
        PTP_CrossfallPrototype = 0x11,
        PTP_LaneSectionPrototype = 0x12,
        PTP_RoadSystemPrototype = 0x14,
        PTP_ScenerySystemPrototype = 0x18,
        PTP_VehicleSystemPrototype = 0x100,
        PTP_PedestrianSystemPrototype = 0x200,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PrototypeManager(QObject *parent);
    virtual ~PrototypeManager();

    // User Prototypes //
    //
    bool loadPrototypes(const QString &fileName);

    void addRoadPrototype(const QString &name, const QIcon &icon, RSystemElementRoad *road, PrototypeManager::PrototypeType type,QString &system,QString &typeName,QString &lanes);
    QList<PrototypeContainer<RSystemElementRoad *> *> getRoadPrototypes(PrototypeManager::PrototypeType type) const
    {
        return roadPrototypes_.values(type);
    }
    
    RSystemElementRoad *getRoadPrototype(PrototypeManager::PrototypeType type,QString typeName);

    void addRoadSystemPrototype(const QString &name, const QIcon &icon, RoadSystem *roadSystem);
    QList<PrototypeContainer<RoadSystem *> *> getRoadSystemPrototypes() const
    {
        return roadSystemPrototypes_;
    }

protected:
private:
    PrototypeManager(); /* not allowed */
    PrototypeManager(const PrototypeManager &); /* not allowed */
    PrototypeManager &operator=(const PrototypeManager &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // User Prototypes //
    //
    QMultiMap<PrototypeManager::PrototypeType, PrototypeContainer<RSystemElementRoad *> *> roadPrototypes_;
    QList<PrototypeContainer<RoadSystem *> *> roadSystemPrototypes_;
};

#endif // PROTOTYPE_HPP
