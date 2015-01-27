/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.06.2010
**
**************************************************************************/

#ifndef PEDESTRIANGROUP_HPP
#define PEDESTRIANGROUP_HPP

#include "src/data/dataelement.hpp"

class PedestrianSystem;

class PedestrianGroup : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum PedestrianGroupChange
    {
        CVG_PedestrianSystemChanged = 0x1,
        CVG_PedestrianChanged = 0x2,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PedestrianGroup();
    virtual ~PedestrianGroup();

    // Pedestrians //
    //
    QMap<QString, Pedestrian *> getPedestrians() const
    {
        return pedestrians_;
    }

    // PedestrianGroup //
    //
    void addPedestrian(Pedestrian *pedestrian);

    // PedestrianSystem //
    //
    PedestrianSystem *getParentPedestrianSystem() const
    {
        return parentPedestrianSystem_;
    }
    void setParentPedestrianSystem(PedestrianSystem *pedestrianSystem);

    // Attributes
    void setSpawnRange(double r)
    {
        spawnRange_ = r;
        spawnRangeSet_ = true;
    }
    double getSpawnRange() const
    {
        return spawnRange_;
    }
    bool hasSpawnRange() const
    {
        return spawnRangeSet_;
    }

    void setMaxPeds(int m)
    {
        maxPeds_ = m;
        maxPedsSet_ = true;
    }
    int getMaxPeds() const
    {
        return maxPeds_;
    }
    bool hasMaxPeds() const
    {
        return maxPedsSet_;
    }

    void setReportInterval(double i)
    {
        reportInterval_ = i;
        reportIntervalSet_ = true;
    }
    double getReportInterval() const
    {
        return reportInterval_;
    }
    bool hasReportInterval() const
    {
        return reportIntervalSet_;
    }

    void setAvoidCount(int c)
    {
        avoidCount_ = c;
        avoidCountSet_ = true;
    }
    int getAvoidCount() const
    {
        return avoidCount_;
    }
    bool hasAvoidCount() const
    {
        return avoidCountSet_;
    }

    void setAvoidTime(double t)
    {
        avoidTime_ = t;
        avoidTimeSet_ = true;
    }
    double getAvoidTime() const
    {
        return avoidTime_;
    }
    bool hasAvoidTime() const
    {
        return avoidTimeSet_;
    }

    void setAutoFiddle(bool a)
    {
        autoFiddle_ = a;
        autoFiddleSet_ = true;
    }
    bool getAutoFiddle() const
    {
        return autoFiddle_;
    }
    bool hasAutoFiddle() const
    {
        return autoFiddleSet_;
    }

    void setMovingFiddle(bool m)
    {
        movingFiddle_ = m;
        movingFiddleSet_ = true;
    }
    bool getMovingFiddle() const
    {
        return movingFiddle_;
    }
    bool hasMovingFiddle() const
    {
        return movingFiddleSet_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPedestrianGroupChanges() const
    {
        return pedestrianGroupChanges_;
    }
    void addPedestrianGroupChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForPedestrians(Visitor *visitor);

private:
    PedestrianGroup(const PedestrianGroup &); /* not allowed */
    PedestrianGroup &operator=(const PedestrianGroup &); /* not allowed */

    int spawnRange_;
    bool spawnRangeSet_;

    int maxPeds_;
    bool maxPedsSet_;

    double reportInterval_;
    bool reportIntervalSet_;

    int avoidCount_;
    bool avoidCountSet_;

    double avoidTime_;
    bool avoidTimeSet_;

    bool autoFiddle_;
    bool autoFiddleSet_;

    bool movingFiddle_;
    bool movingFiddleSet_;

    //################//
    // PROPERTIES     //
    //################//

private:
    // PedestrianSystem //
    //
    PedestrianSystem *parentPedestrianSystem_;

    // Change flags //
    //
    int pedestrianGroupChanges_;

    // Pedestrians //
    //
    QMap<QString, Pedestrian *> pedestrians_; // owned
};

#endif // PEDESTRIANGROUP_HPP
