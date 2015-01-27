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

#ifndef RSYSTEMELEMENTJUNCTION_HPP
#define RSYSTEMELEMENTJUNCTION_HPP

#include "rsystemelement.hpp"

class JunctionConnection;

//####################//
// JunctionElement    //
//####################//

class RSystemElementJunction : public RSystemElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum JunctionChange
    {
        CJN_ConnectionChanged = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RSystemElementJunction(const QString &name, const QString &id);
    virtual ~RSystemElementJunction();

    // JunctionConnections //
    //
    QMultiMap<QString, JunctionConnection *> getConnections() const
    {
        return connections_;
    }
    QList<JunctionConnection *> getConnections(const QString &incomingRoad) const;
    QList<JunctionConnection *> getConnectingRoadConnections(const QString &connectingRoad) const;
    JunctionConnection *getConnection(const QString &incomingRoad, const QString &connectingRoad) const;
    void addConnection(JunctionConnection *connection);
    void delConnection(JunctionConnection *connection);
    void delConnections();

    void checkConnectionIds(const QMap<QString, QString> &roadIds);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getJunctionChanges() const
    {
        return junctionChanges_;
    }
    void addJunctionChanges(int changes);

    // Prototype Pattern //
    //
    RSystemElementJunction *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForConnections(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);

private:
    RSystemElementJunction(); /* not allowed */
    RSystemElementJunction(const RSystemElementJunction &); /* not allowed */
    RSystemElementJunction &operator=(const RSystemElementJunction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change //
    //
    int junctionChanges_;

    // JunctionConnections //
    //
    QMultiMap<QString, JunctionConnection *> connections_; // owned
};

#endif // RSYSTEMELEMENTJUNCTION_HPP
