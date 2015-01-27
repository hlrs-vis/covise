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

#ifndef ELEVATIONEDITOR_HPP
#define ELEVATIONEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;

class TopviewGraph;
class ProfileGraph;

class RSystemElementRoad;

class SectionHandle;
class ElevationMoveHandle;

class RoadSystemItem;

class ElevationRoadSystemItem;
class ElevationRoadItem;
class ElevationSectionItem;
class ElevationRoadPolynomialItem;

#include <QPointF>
#include <QMap>
#include <QMultiMap>
#include <QRectF>

class ElevationEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph, ProfileGraph *profileGraph);
    virtual ~ElevationEditor();

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
    QMap<RSystemElementRoad *, ElevationRoadPolynomialItem *> getSelectedElevationItems()
    {
        return selectedElevationRoadItems_;
    };

    // Selected Roads //
    //
    void addSelectedRoad(ElevationRoadPolynomialItem *roadItem);
    int delSelectedRoad(RSystemElementRoad *road);
    void insertSelectedRoad(RSystemElementRoad *road);
    void initBox();
    void fitView();

    // MoveHandles //
    //
    void registerMoveHandle(ElevationMoveHandle *handle);
    int unregisterMoveHandle(ElevationMoveHandle *handle);
    bool translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    //	virtual void			mouseAction(MouseAction * mouseAction);
    //	virtual void			keyAction(KeyAction * keyAction);

protected:
    virtual void init();
    virtual void kill();

private:
    ElevationEditor(); /* not allowed */
    ElevationEditor(const ElevationEditor &); /* not allowed */
    ElevationEditor &operator=(const ElevationEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // Graph //
    //
    ElevationRoadSystemItem *roadSystemItem_;

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
    QMap<RSystemElementRoad *, ElevationRoadPolynomialItem *> selectedElevationRoadItems_;
    QMultiMap<int, ElevationMoveHandle *> selectedMoveHandles_;

    // Bounding Box for all selected roads //
    //
    QRectF boundingBox_;
    qreal xtrans_;
};

#endif // ELEVATIONEDITOR_HPP
