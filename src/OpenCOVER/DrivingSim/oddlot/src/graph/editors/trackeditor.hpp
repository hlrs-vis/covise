/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11.03.2010
**
**************************************************************************/

#ifndef TRACKEDITOR_HPP
#define TRACKEDITOR_HPP

#include "projecteditor.hpp"

#include <QMultiMap>
#include <QPointF>

class ProjectData;
class TopviewGraph;

class RoadSystem;
class RSystemElementRoad;
class TrackComponent;

class TrackMoveHandle;
class TrackRotateHandle;
class TrackAddHandle;

class RoadMoveHandle;
class RoadRotateHandle;

class QGraphicsLineItem;
class CircularRotateHandle;

class TrackRoadSystemItem;

// TODO
class SectionHandle;

class TrackEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

private:
    enum TrackEditorState
    {
        STE_NONE,
        STE_NEW_PRESSED,
        STE_ROADSYSTEM_ADD
    };

    enum TransformType
    {
        TT_MOVE = 1,
        TT_ROTATE = 2
    };

    struct TrackMoveProperties
    {
        TrackComponent *highSlot;
        TrackComponent *lowSlot;
        QPointF dPos;
        double heading;
        short int transform;
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~TrackEditor();

    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    virtual void mouseAction(MouseAction *mouseAction);
    virtual void keyAction(KeyAction *keyAction);

    bool translateTrack(TrackComponent *track, const QPointF &pressPos, const QPointF &mousePos);

    // MoveHandles //
    //
    void registerTrackMoveHandle(TrackMoveHandle *handle);
    int unregisterTrackMoveHandle(TrackMoveHandle *handle);
    bool translateTrackMoveHandles(const QPointF &pressPos, const QPointF &mousePos);
    bool translateTrackComponents(const QPointF &pressPos, const QPointF &mousePos);
    bool validate(TrackMoveProperties *props);
    void translate(TrackMoveProperties *props);

    // RoadMoveHandles //
    //
    void registerRoadMoveHandle(RoadMoveHandle *handle);
    int unregisterRoadMoveHandle(RoadMoveHandle *handle);
    bool translateRoadMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    // RoadRotateHandles //
    //
    void registerRoadRotateHandle(RoadRotateHandle *handle);
    int unregisterRoadRotateHandle(RoadRotateHandle *handle);
    bool rotateRoadRotateHandles(const QPointF &pivotPoint, double angleDegrees);

    // AddHandles //
    //
    void registerTrackAddHandle(TrackAddHandle *handle);
    int unregisterTrackAddHandle(TrackAddHandle *handle);

    // Register Roads //
    void registerRoad(RSystemElementRoad *road);

#if 0
	// RotateHandles //
	//
	void						registerTrackRotateHandle(TrackRotateHandle * handle);
	int						unregisterTrackRotateHandle(TrackRotateHandle * handle);
	double					rotateTrackRotateHandles(double dHeading, double globalHeading);
#endif

    // Section Handle //
    //
    SectionHandle *getSectionHandle() const;

protected:
    virtual void init();
    virtual void kill();

private:
    TrackEditor(); /* not allowed */
    TrackEditor(const TrackEditor &); /* not allowed */
    TrackEditor &operator=(const TrackEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadSystem //
    //
    TrackRoadSystemItem *trackRoadSystemItem_;

    // MouseAction //
    //
    QPointF pressPoint_;

    // New Road Tool //
    //
    QGraphicsLineItem *newRoadLineItem_;

    // Add RoadSystem Tool //
    //
    CircularRotateHandle *addRoadSystemHandle_;

    // Move/Add/Rotate Tool //
    //
    QMultiMap<int, TrackMoveHandle *> selectedTrackMoveHandles_;
    QMultiMap<int, TrackAddHandle *> selectedTrackAddHandles_;
    //QMultiMap<double, TrackRotateHandle *>	selectedTrackRotateHandles_;

    QMultiMap<int, RoadMoveHandle *> selectedRoadMoveHandles_;
    QMultiMap<int, RoadRotateHandle *> selectedRoadRotateHandles_;

    QList<RSystemElementRoad *> selectedRoads_;

    // Add Tool //
    //
    RSystemElementRoad *currentRoadPrototype_;
    RoadSystem *currentRoadSystemPrototype_;

    // Move Tile Tool //
    //
    QString currentTile_;

    // State //
    //
    TrackEditor::TrackEditorState state_;

    // TODO
    SectionHandle *sectionHandle_;
};

#endif // TRACKEDITOR_HPP
