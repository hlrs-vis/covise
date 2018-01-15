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
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/odrID.hpp"

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

	enum ContactPointValue
	{
		JCP_NONE,
		JCP_START,
		JCP_END
	};

	static ContactPointValue parseContactPoint(const QString &value);
	static QString parseContactPointBack(ContactPointValue value);

    struct ConnectionUserData
    {
        double numerator;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionConnection(const QString &id, const odrID &incomingRoad, const odrID &connectingRoad, JunctionConnection::ContactPointValue contactPoint, double numerator);
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
	const QString &getId() const
    {
        return id_;
    }
    void setId(const QString &id);

	const odrID & getIncomingRoad() const
    {
        return incomingRoad_;
    }
    void setIncomingRoad(const odrID &id);

    const odrID &getConnectingRoad() const
    {
        return connectingRoad_;
    }
    void setConnectingRoad(const odrID &id);

	ContactPointValue getContactPoint() const
    {
        return contactPoint_;
    }
    void setContactPoint(ContactPointValue contactPoint);

    double getNumerator() const
    {
        return userData_.numerator;
    }
    void setNumerator(double numerator);

    // LaneLinks //
    //
    void addLaneLink(int from, int to);
    void setLaneLinks(const QMap<int, int> laneLinks)
    {
        laneLinks_ = laneLinks;
    }

    void removeLaneLink(int from);
    void removeLaneLinks();
    QMap<int, int> getLaneLinks() const
    {
        return laneLinks_;
    }

    int getFromLane(int to)
    {
        return laneLinks_.key(to, Lane::NOLANE);
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
    odrID incomingRoad_;
    odrID connectingRoad_;
	ContactPointValue contactPoint_;
    ConnectionUserData userData_;

    // LaneLinks //
    //
    QMap<int, int> laneLinks_;
};

#endif // JUNCTIONCONNECTION_HPP
