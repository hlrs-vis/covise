/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/25/2010
**
**************************************************************************/

#ifndef JUNCTIONITEM_HPP
#define JUNCTIONITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RSystemElementJunction;
class RSystemElementJunction;
class RoadSystemItem;
class TextHandle;

class JunctionItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionItem(RoadSystemItem *roadSystemItem, RSystemElementJunction *junction);
    virtual ~JunctionItem();

    // Road //
    //
    RSystemElementJunction *getJunction() const
    {
        return junction_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    JunctionItem(); /* not allowed */
    JunctionItem(const JunctionItem &); /* not allowed */
    JunctionItem &operator=(const JunctionItem &); /* not allowed */

    void init();
    void updatePath();
    void updatePathList();

    //################//
    // SLOTS          //
    //################//

public slots:
    bool removeJunction();
    void addToCurrentTile();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

private:
    // Junction/Roads //
    //
    RSystemElementJunction *junction_;
    QList<RSystemElementRoad *> paths_;

    // RoadSystem //
    //
    RoadSystemItem *roadSystemItem_;

    // Text //
    //
    TextHandle *textHandle_;
};

#endif // JUNCTIONITEM_HPP
