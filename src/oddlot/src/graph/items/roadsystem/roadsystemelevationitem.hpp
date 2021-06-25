/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.06.2010
**
**************************************************************************/

#ifndef ROADSYSTEMELEVATIONITEM_HPP
#define ROADSYSTEMELEVATIONITEM_HPP

#include "src/graph/items/graphelement.hpp"

#include <QMap>

class RoadSystem;

class RoadSystemElevationItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSystemElevationItem(ProjectGraph *projectGraph, RoadSystem *roadSystem);
    virtual ~RoadSystemElevationItem();

    // ProjectGraph //
    //
    virtual ProjectGraph *getProjectGraph() const
    {
        return projectGraph_;
    }

    // RoadSystem //
    //
    RoadSystem *getRoadSystem() const
    {
        return roadSystem_;
    }

    // MapItems //
    //
    //	void						loadMap(const QString & filename, const QPointF & pos);
    //	void						deleteMap();
    //	void						lockMaps(bool locked);
    //	void						setMapOpacity(double opacity);
    //	void						setMapWith(double width, bool keepRatio);
    //	void						setMapHeight(double height, bool keepRatio);

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    RoadSystemElevationItem(); /* not allowed */
    RoadSystemElevationItem(const RoadSystemElevationItem &); /* not allowed */
    RoadSystemElevationItem &operator=(const RoadSystemElevationItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    // ProjectGraph //
    //
    ProjectGraph *projectGraph_;

    // RoadSystem //
    //
    RoadSystem *roadSystem_;

    // MapItems //
    //
    //	QMap<QString, SceneryMapItem *>		mapItems_;
};

#endif // ROADSYSTEMELEVATIONITEM_HPP
