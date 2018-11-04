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

class ProjectData;
class TopviewGraph;

class LaneRoadSystemItem;
class BaseLaneMoveHandle;

class SectionHandle;
class PointHandle;

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
	PointHandle *getAddWidthHandle() const;

    // Tool //
    //
    virtual void toolAction(ToolAction *);
	virtual void mouseAction(MouseAction *mouseAction);

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

    // Handles //
    //
    SectionHandle *insertSectionHandle_;
	PointHandle *pointHandle_;

	// Edit Mode (width or border) //
	//
//	bool borderEditMode_;

	QList<BaseLaneMoveHandle *> selectedLaneMoveHandles_;
};

#endif // LANEEDITOR_HPP
