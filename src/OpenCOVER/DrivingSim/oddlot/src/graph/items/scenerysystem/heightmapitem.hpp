/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#ifndef HEIGHTMAPITEM_HPP
#define HEIGHTMAPITEM_HPP

#include "scenerymapitem.hpp"

class Heightmap;
class ScenerySystemItem;

class HeightmapItem : public SceneryMapItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit HeightmapItem(ScenerySystemItem *parentScenerySystem, Heightmap *heightmap);
    virtual ~HeightmapItem();

    Heightmap *getHeightmap() const
    {
        return heightmap_;
    }

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    HeightmapItem(); /* not allowed */
    HeightmapItem(const HeightmapItem &); /* not allowed */
    HeightmapItem &operator=(const HeightmapItem &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeMap();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);

    //################//
    // PROPERTIES     //
    //################//

private:
    Heightmap *heightmap_;
};

#endif // HEIGHTMAPITEM_HPP
