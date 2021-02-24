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

#ifndef ROADLINKEDITOR_HPP
#define ROADLINKEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;
class TopviewGraph;

class RoadLinkRoadSystemItem;
class RSystemElementRoad;
class LaneSection;
class Lane;

class QGraphicsItem;

class RoadLinkEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~RoadLinkEditor();

    // Tool //
    //
    virtual void toolAction(ToolAction *);
	virtual void mouseAction(MouseAction *mouseAction);

    void assignParameterSelection(ODD::ToolId id);
	void clearToolObjectSelection();

protected:
    virtual void init();
    virtual void kill();

private:
    RoadLinkEditor(); /* not allowed */
    RoadLinkEditor(const RoadLinkEditor &); /* not allowed */
    RoadLinkEditor &operator=(const RoadLinkEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:
	// Parameter Settings //
	//
	virtual void apply();
	virtual void reject();
	virtual void reset();

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadSystem //
    //
    RoadLinkRoadSystemItem *roadSystemItem_;

    void removeZeroWidthLanes(RSystemElementRoad * road); 

	// List of selected roads //
	//
	QList<RSystemElementRoad *> selectedRoads_;

	// Selected handles //
	//
	QGraphicsItem *linkItem_;
	QGraphicsItem *sinkItem_;

	//Threshold for the linking roads
	//
	double threshold_;

	// Currently selected Parameter //
	//
	int currentParamId_;

	// necessary selected elements to make APPLY visible //
	//
	int applyCount_;
};

#endif // ROADLINKEDITOR_HPP
