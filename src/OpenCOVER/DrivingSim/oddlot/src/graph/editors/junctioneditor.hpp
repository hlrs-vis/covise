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

#ifndef JUNCTIONEDITOR_HPP
#define JUNCTIONEDITOR_HPP

#include "projecteditor.hpp"

#include <QMultiMap>
#include <QPointF>

class ProjectData;
class TopviewGraph;
class ProfileGraph;

class RoadSystem;
class RSystemElementRoad;
class RSystemElementJunction;
class TrackComponent;
class Lane;
class LaneSection;

class JunctionMoveHandle;
class JunctionAddHandle;
class JunctionLaneWidthMoveHandle;

class QGraphicsLineItem;
class CircularRotateHandle;

class JunctionRoadSystemItem;
class JunctionLaneRoadSystemItem;

// TODO
class SectionHandle;

class JunctionEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

private:
    enum JunctionEditorState
    {
        STE_NONE,
        STE_NEW_PRESSED,
        STE_ROADSYSTEM_ADD
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~JunctionEditor();

    // Tool, Mouse & Key //
    //
    virtual void toolAction(ToolAction *toolAction);
    virtual void mouseAction(MouseAction *mouseAction);
    virtual void keyAction(KeyAction *keyAction);

    bool translateTrack(TrackComponent *track, const QPointF &pressPos, const QPointF &mousePos);

    // MoveHandles //
    //
    void registerJunctionMoveHandle(JunctionMoveHandle *handle);
    int unregisterJunctionMoveHandle(JunctionMoveHandle *handle);
    bool translateJunctionMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    void registerLaneMoveHandle(JunctionLaneWidthMoveHandle *handle);
    int unregisterLaneMoveHandle(JunctionLaneWidthMoveHandle *handle);
    bool translateLaneMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

    // AddHandles //
    //
    void registerJunctionAddHandle(JunctionAddHandle *handle);
    int unregisterJunctionAddHandle(JunctionAddHandle *handle);

    // AddLanes //
    //
    void registerLane(Lane *lane);
    void registerRoad(RSystemElementRoad *road);

#if 0
	// RotateHandles //
	//
	void						registerJunctionRotateHandle(JunctionRotateHandle * handle);
	int						unregisterJunctionRotateHandle(JunctionRotateHandle * handle);
	double					rotateJunctionRotateHandles(double dHeading, double globalHeading);
#endif

    // Section Handle //
    //
    SectionHandle *getSectionHandle() const;

    ProfileGraph *getProfileGraph()
    {
        return profileGraph_;
    };

    double getThreshold()
    {
        return threshold_;
    };

    // Create connecting road in junction //
    //
    void createRoad(QList<Lane *>);
    void createRoad(QList<RSystemElementRoad *>);
    RSystemElementRoad *createSpiral(RSystemElementRoad *road1, RSystemElementRoad *road2, bool startContact1, bool startContact2, double offset1 = 0.0, double offset2 = 0.0);

    double widthOffset(RSystemElementRoad *road, Lane *lane, LaneSection *laneSection, double s, bool addOwnLaneWidth); // calculates the offset of a lane from the center of the road

protected:
    virtual void init();
    virtual void kill();

private:
    JunctionEditor(); /* not allowed */
    JunctionEditor(const JunctionEditor &); /* not allowed */
    JunctionEditor &operator=(const JunctionEditor &); /* not allowed */

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
    JunctionRoadSystemItem *junctionRoadSystemItem_;

    // ProfileGraph //
    //
    ProfileGraph *profileGraph_;

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
    QMultiMap<int, JunctionMoveHandle *> selectedJunctionMoveHandles_;
    QMultiMap<int, JunctionAddHandle *> selectedJunctionAddHandles_;

    // LaneWidth //
    //
    QMultiMap<int, JunctionLaneWidthMoveHandle *> selectedLaneMoveHandles_;

    // State //
    //
    JunctionEditor::JunctionEditorState state_;

    // TODO
    SectionHandle *sectionHandle_;

    // Lanes //
    //
    JunctionLaneRoadSystemItem *laneRoadSystemItem_;

    // Map of pairs of incoming lanes //
    //
    QList<Lane *> selectedLanes_;

    // Map of pairs of incoming roads //
    //
    QList<RSystemElementRoad *> selectedRoads_;

    // Selected Junction //
    //
    RSystemElementJunction *junction_;

    //Threshold for the cutting circle
    //
    double threshold_;
};

#endif // TRACKEDITOR_HPP
