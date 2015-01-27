/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/19/2010
**
**************************************************************************/

#ifndef BOUNDINGBOXVISITOR_HPP
#define BOUNDINGBOXVISITOR_HPP

#include "src/data/acceptor.hpp"

#include <QRectF>

class BoundingBoxVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit BoundingBoxVisitor();
    virtual ~BoundingBoxVisitor()
    { /* does nothing */
    }

    QRectF getBoundingBox() const
    {
        return boundingBox_;
    }

    // Visitor Pattern //
    //
    virtual void visit(RoadSystem *roadSystem);
    virtual void visit(RSystemElementRoad *road);

    virtual void visit(TrackComposite *acceptor);

    virtual void visit(TrackElement *acceptor);
    //	virtual void			visit(TrackElementLine * acceptor);
    //	virtual void			visit(TrackElementArc * acceptor);
    //	virtual void			visit(TrackElementSpiral * acceptor);
    //	virtual void			visit(TrackElementPoly3 * acceptor);

    virtual void visit(ScenerySystem *acceptor);
    virtual void visit(SceneryMap *acceptor);

private:
    //BoundingBoxVisitor(); /* not allowed */
    BoundingBoxVisitor(const BoundingBoxVisitor &); /* not allowed */
    BoundingBoxVisitor &operator=(const BoundingBoxVisitor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    QRectF boundingBox_;
};

#endif // BOUNDINGBOXVISITOR_HPP
