/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#ifndef ROADLINK_HPP
#define ROADLINK_HPP

#include "src/data/dataelement.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/odrID.hpp"

class RoadLink : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum RSystemElementChange
    {
        CRL_IdChanged = 0x1,
        CRL_TypeChanged = 0x2,
        CRL_ContactPointChanged = 0x4,
        CRL_ParentRoadChanged = 0x8
    };

    enum RoadLinkType
    {
        DRL_PREDECESSOR,
        DRL_SUCCESSOR,
        DRL_UNKNOWN
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLink(const QString &elementType, const odrID &elementId, JunctionConnection::ContactPointValue contactPoint);
    virtual ~RoadLink();

    // RoadLink //
    //
    QString getElementType() const
    {
        return elementType_;
    }
    void setElementType(const QString &elementType);

	const odrID &getElementId() const
    {
        return elementId_;
    }
    void setElementId(const odrID &elementId);

	JunctionConnection::ContactPointValue getContactPoint() const
    {
        return contactPoint_;
    }
	QString getContactPointString() const;
    void setContactPoint(JunctionConnection::ContactPointValue contactPoint);

    RoadLinkType getRoadLinkType() const
    {
        return type_;
    }

    // Road //
    //
    RSystemElementRoad *getParentRoad() const
    {
        return parentRoad_;
    }
    void setParentRoad(RSystemElementRoad *road, RoadLinkType type);

    bool isLinkValid() const;

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getRoadLinkChanges() const
    {
        return roadLinkChange_;
    }
    void addRoadLinkChanges(int changes);

    // Prototype Pattern //
    //
    RoadLink *getClone() const
    {
        return new RoadLink(elementType_, elementId_, contactPoint_);
    }

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
    RoadLink(); /* not allowed */
    RoadLink(const RoadLink &); /* not allowed */
    RoadLink &operator=(const RoadLink &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

    //public slots:

    //public signals:

    //################//
    // PROPERTIES     //
    //################//

private:
    // Observer Pattern //
    //
    int roadLinkChange_;

    // Parent //
    //
    RSystemElementRoad *parentRoad_;

    RoadLinkType type_;

    // RoadLink //
    //
    QString elementType_; // "road" or "junction"
	odrID elementId_; // ID of the linked road
	JunctionConnection::ContactPointValue contactPoint_; // contact point of the linked element ("start" or "end")
};

#endif // ROADLINK_HPP
