/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#ifndef LANEEDITOR_HPP
#define LANEEDITOR_HPP

#include "projecteditor.hpp"

#include "src/graph/profilegraph.hpp"
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"
#include "src/graph/items/handles/lanemovehandle.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

class ProjectData;
class TopviewGraph;

class LaneWidth;
class LaneWidthMoveHandle;
class LaneRoadSystemItem;
class RoadSystemItem;

class SectionHandle;

class LaneEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit LaneEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~LaneEditor();

    // Handle //
    //
    SectionHandle *getInsertSectionHandle() const;

    // Tool //
    //
    virtual void toolAction(ToolAction *);
	virtual void mouseAction(MouseAction *mouseAction);


    // MoveHandles //
    //
    void registerMoveHandle(LaneWidthMoveHandle *handle);
    int unregisterMoveHandle(LaneWidthMoveHandle *handle);
    void setWidth(double w);
    bool translateMoveHandles(const QPointF &pressPos, const QPointF &mousePos);

	void removeMoveHandle();

	// BorderMoveHandles //
	//

	void registerMoveHandle(BaseLaneMoveHandle *handle);
	int unregisterMoveHandle(BaseLaneMoveHandle *handle);


	bool translateLaneBorder(const QPointF &pressPos, const QPointF &mousePos, double width = 0.0, bool setWidth = false);

protected:
    virtual void init();
    virtual void kill();

private:
    LaneEditor(); /* not allowed */
    LaneEditor(const LaneEditor &); /* not allowed */
    LaneEditor &operator=(const LaneEditor &); /* not allowed */

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
    LaneRoadSystemItem *roadSystemItem_;

    // Handle //
    //
    SectionHandle *insertSectionHandle_;

	// Edit Mode (width or border) //
	//
//	bool borderEditMode_;

    QMultiMap<int, LaneWidthMoveHandle *> selectedMoveHandles_;
	QList<BaseLaneMoveHandle *> selectedLaneMoveHandles_;
};

#endif // LANEEDITOR_HPP
