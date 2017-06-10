/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.06.2010
**
**************************************************************************/

#ifndef SCENERYSYSTEMITEM_HPP
#define SCENERYSYSTEMITEM_HPP

#include "src/graph/items/graphelement.hpp"

#include <QMap>

class ScenerySystem;
class SceneryMapItem;

class ScenerySystemItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ScenerySystemItem(TopviewGraph *topviewGraph, ScenerySystem *scenerySystem);
    virtual ~ScenerySystemItem();

    // TopviewGraph //
    //
    virtual TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }

    // ScenerySystem //
    //
    ScenerySystem *getScenerySystem() const
    {
        return scenerySystem_;
    }

    // MapItems //
    //
    void loadMap(const QString &filename, const QPointF &pos);
    void loadGoogleMap(const QString &filename, double mapPosLat, double mapPosLon);
    void deleteMap();
    void lockMaps(bool locked);
    void setMapOpacity(double opacity);
    void setMapX(double x);
    void setMapY(double y);
    void setMapWith(double width, bool keepRatio);
    void setMapHeight(double height, bool keepRatio);

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    ScenerySystemItem(); /* not allowed */
    ScenerySystemItem(const ScenerySystemItem &); /* not allowed */
    ScenerySystemItem &operator=(const ScenerySystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // TopviewGraph //
    //
    TopviewGraph *topviewGraph_;

    // ScenerySystem //
    //
    ScenerySystem *scenerySystem_;

    // MapItems //
    //
    QMap<QString, SceneryMapItem *> mapItems_;
};

#endif // SCENERYSYSTEMITEM_HPP
