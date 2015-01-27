/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   6/11/2010
**
**************************************************************************/

#ifndef MAPITEM_HPP
#define MAPITEM_HPP

#include "src/graph/items/graphelement.hpp"

class SceneryMap;
class ScenerySystemItem;

class SceneryMapItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SceneryMapItem(ScenerySystemItem *parentScenerySystem, SceneryMap *sceneryMap);
    virtual ~SceneryMapItem();

    bool isLoaded() const
    {
        return loaded_;
    }

    bool isLocked() const
    {
        return loaded_;
    }
    void setLocked(bool locked);
    void setMapX(double x);
    void setMapY(double y);
    void setMapWidth(double width, bool keepRatio);
    void setMapHeight(double height, bool keepRatio);

    SceneryMap *getMap() const
    {
        return sceneryMap_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest();

private:
    SceneryMapItem(); /* not allowed */
    SceneryMapItem(const SceneryMapItem &); /* not allowed */
    SceneryMapItem &operator=(const SceneryMapItem &); /* not allowed */

    bool loadFile();
    void updateSize();
    void updatePosition();
    void updateOpacity();
    void updateFilename();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeMap();

    //################//
    // EVENTS         //
    //################//

public:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    //################//
    // PROPERTIES     //
    //################//

private:
    SceneryMap *sceneryMap_;

    QGraphicsPixmapItem *pixmapItem_;

    bool loaded_;
    bool locked_;
};

#endif // MAPITEM_HPP
