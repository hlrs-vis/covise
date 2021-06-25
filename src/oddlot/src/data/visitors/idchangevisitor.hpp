/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.07.2010
**
**************************************************************************/

#ifndef IDCHANGEVISITOR_HPP
#define IDCHANGEVISITOR_HPP

#include "src/data/acceptor.hpp"
#include "src/data/roadsystem/odrID.hpp"

#include <QMap>

class IdChangeVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit IdChangeVisitor(const QMap<odrID, odrID> &roadIds, const QMap<odrID, odrID> &controllerIds, const QMap<odrID, odrID> &junctionIds, const QMap<odrID, odrID> &fiddleyardIds);
    virtual ~IdChangeVisitor()
    { /* does nothing */
    }

    // Visitor Pattern //
    //
    //	virtual void			visit(RoadSystem * roadSystem);
    virtual void visit(RSystemElementRoad *road);
    virtual void visit(RSystemElementController *controller);
    virtual void visit(RSystemElementJunction *junction);
    virtual void visit(RSystemElementFiddleyard *fiddleyard);

    virtual void visit(JunctionConnection *connection);

private:
    IdChangeVisitor(); /* not allowed */
    IdChangeVisitor(const IdChangeVisitor &); /* not allowed */
    IdChangeVisitor &operator=(const IdChangeVisitor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    QMap<odrID, odrID> roadIds_;
    QMap<odrID, odrID> controllerIds_;
    QMap<odrID, odrID> junctionIds_;
    QMap<odrID, odrID> fiddleyardIds_;
};
#endif // IDCHANGEVISITOR_HPP
