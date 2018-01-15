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

#include <QMap>

class IdChangeVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit IdChangeVisitor(const QMap<QString, QString> &roadIds, const QMap<QString, QString> &controllerIds, const QMap<QString, QString> &junctionIds, const QMap<QString, QString> &fiddleyardIds);
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
    QMap<QString, QString> roadIds_;
    QMap<QString, QString> controllerIds_;
    QMap<QString, QString> junctionIds_;
    QMap<QString, QString> fiddleyardIds_;
};
#endif // IDCHANGEVISITOR_HPP
