/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/5/2010
**
**************************************************************************/

#ifndef JUNCTIONCONNECTION_HPP
#define JUNCTIONCONNECTION_HPP

#include "src/data/dataelement.hpp"

// TODO: Observer Pattern

class JunctionConnection : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum JunctionConnectionChange
    {
        CJC_ParentJunctionChanged = 0x1,
        CJC_IdChanged = 0x2,
        CJC_IncomingRoadChanged = 0x4,
        CJC_ConnectingRoadChanged = 0x8,
        CJC_ContactPointChanged = 0x10,
        CJC_NumeratorChanged = 0x20,
        CJC_LaneLinkChanged = 0x40
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionConnection(const QString &id, const QString &incomingRoad, const QString &connectingRoad, const QString &contactPoint, double numerator);
    virtual ~JunctionConnection()
    { /* does nothing */
    }

    // Junction //
    //
    RSystemElementJunction *getParentJunction() const
    {
        return parentJunction_;
    }
    void setParentJunction(RSystemElementJunction *parentJunction);

    // JunctionConnection //
    //
    QString getId() const
    {
        return id_;
    }
    void setId(const QString &id);

    QString getIncomingRoad() const
    {
        return incomingRoad_;
    }
    void setIncomingRoad(const QString &id);

    QString getConnectingRoad() const
    {
        return connectingRoad_;
    }
    void setConnectingRoad(const QString &id);

    QString getContactPoint() const
    {
        return contactPoint_;
    }
    void setContactPoint(const QString &contactPoint);

    double getNumerator() const
    {
        return numerator_;
    }
    void setNumerator(double numerator);

    // LaneLinks //
    //
    void addLaneLink(int from, int to);
    QMap<int, int> getLaneLinks() const
    {
        return laneLinks_;
    }

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getJunctionConnectionChanges() const
    {
        return junctionConnectionChanges_;
    }
    void addJunctionConnectionChanges(int changes);

    // Prototype Pattern //
    //
    JunctionConnection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    JunctionConnection(); /* not allowed */
    JunctionConnection(const JunctionConnection &); /* not allowed */
    JunctionConnection &operator=(const JunctionConnection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change //
    //
    int junctionConnectionChanges_;

    // Junction //
    //
    RSystemElementJunction *parentJunction_;

    // JunctionConnection //
    //
    QString id_;
    QString incomingRoad_;
    QString connectingRoad_;
    QString contactPoint_;
    double numerator_;

    // LaneLinks //
    //
    QMap<int, int> laneLinks_;
};

#endif // JUNCTIONCONNECTION_HPP
