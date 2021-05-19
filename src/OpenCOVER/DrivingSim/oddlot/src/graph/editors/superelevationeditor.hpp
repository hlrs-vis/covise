/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   16.07.2010
 **
 **************************************************************************/

#ifndef SUPERELEVATIONEDITOR_HPP
#define SUPERELEVATIONEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;

class TopviewGraph;
class ProfileGraph;

class RSystemElementRoad;

class SectionHandle;
class SuperelevationMoveHandle;

class RoadSystemItem;

class SuperelevationRoadSystemItem;
class SuperelevationRoadItem;
class SuperelevationSectionItem;
class SuperelevationRoadPolynomialItem;

#include <QPointF>
#include <QMap>
#include <QMultiMap>
#include <QRectF>

class SuperelevationEditor : public ProjectEditor
{
    Q_OBJECT

        //################//
        // FUNCTIONS      //
        //################//

public:
    explicit SuperelevationEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph);
    virtual ~SuperelevationEditor();

    // TODO look for better solution
    SectionHandle *getInsertSectionHandle();

    // ProfileGraph //
    //
    ProfileGraph *getProfileGraph() const
    {
        return profileGraph_;
    }

    // Smooth Radius //
    //
    double getSmoothRadius() const
    {
        return smoothRadius_;
    }

    //Get selected roads //
    //
    SuperelevationRoadPolynomialItem *getSelectedSuperelevationItem()
    {
        return selectedSuperelevationItem_;
    };

    // Selected Roads //
    //
    void addSelectedRoad(SuperelevationRoadPolynomialItem *roadItem);
    void delSelectedRoad(RSystemElementRoad *road);
    void insertSelectedRoad(RSystemElementRoad *road);
    void initBox();
    void fitView();

    // MoveHandles //
    //
    void registerMoveHandle(SuperelevationMoveHandle *handle);
    int unregisterMoveHandle(SuperelevationMoveHandle *handle);
    bool translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    // virtual void   mouseAction(MouseAction * mouseAction);
    // virtual void   keyAction(KeyAction * keyAction);

protected:
    virtual void init();
    virtual void kill();

private:
    SuperelevationEditor(); /* not allowed */
    SuperelevationEditor(const SuperelevationEditor &); /* not allowed */
    SuperelevationEditor &operator=(const SuperelevationEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:
    // Parameter Settings //
    //
    virtual void apply() {};
    virtual void reject() {};
    virtual void reset() {};

    //################//
    // PROPERTIES     //
    //################//

private:
    // Graph //
    //
    SuperelevationRoadSystemItem *roadSystemItem_;

    // ProfileGraph //
    //
    ProfileGraph *profileGraph_;
    RoadSystemItem *roadSystemItemPolyGraph_;

    // TODO look for better solution
    SectionHandle *insertSectionHandle_;

    // Smooth Radius //
    //
    double smoothRadius_;

    // ProfileGraph: Selected Items //
    //
    QList<RSystemElementRoad *> selectedRoads_;
    SuperelevationRoadPolynomialItem *selectedSuperelevationItem_;

    QMultiMap<int, SuperelevationMoveHandle *> selectedMoveHandles_;

    // Bounding Box for all selected roads //
    //
    QRectF boundingBox_;
    qreal xtrans_;
};

#endif // SUPERELEVATIONEDITOR_HPP
