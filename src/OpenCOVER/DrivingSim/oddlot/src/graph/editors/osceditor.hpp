/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#ifndef OPENSCENARIOEDITOR_HPP
#define OPENSCENARIOEDITOR_HPP

#include "projecteditor.hpp"

#include <QMultiMap>

class ProjectData;
class TopviewGraph;
class RSystemElementRoad;
class CatalogTreeWidget;

namespace OpenScenario
{
class OpenScenarioBase;
class oscObjectBase;
class oscObject;
}

class OSCBase;
class OSCRoadSystemItem;

class OpenScenarioEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OpenScenarioEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~OpenScenarioEditor();

    // Tool, Mouse & Key //
    //
    virtual void mouseAction(MouseAction *mouseAction);

    // Handle //
    //
 //   OSCHandle *getInsertOSCHandle() const;

    // Tool //
    //
    virtual void toolAction(ToolAction *);

    // Move Signal //
    //
	RSystemElementRoad *findClosestRoad(const QPointF &to, double &s, double &dist, QVector2D &vec);
	bool translateObject(OpenScenario::oscObject * object, RSystemElementRoad * newRoad, QPointF &to);


	// Catalog dock widget changed //
	//
	void catalogChanged(OpenScenario::oscObjectBase *object);

	// New Object with properties chosen in SignalTreeWidget //
	//
//	Object *addObjectToRoad(RSystemElementRoad *road, double s, double t);


protected:
    virtual void init();
    virtual void kill();

private:
    OpenScenarioEditor(); /* not allowed */
    OpenScenarioEditor(const OpenScenarioEditor &); /* not allowed */
    OpenScenarioEditor &operator=(const OpenScenarioEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:

    // Handle //
    //
  //  OSCHandle *insertOSCHandle_;

	// RoadSystem //
	//
	 OSCRoadSystemItem * oscRoadSystemItem_;

	// MainWindow //
	//
	MainWindow * mainWindow_;

	// OpenScenarioBase //
	//
	OSCBase * oscBase_;

	// Selected catalog //
	//
	OpenScenario::oscObjectBase *oscCatalog_;

	ODD::ToolId lastTool_;

	QString catalogElement_;

	// Window with catalog entries //
	//
	CatalogTreeWidget *catalogTree_;

    // RoadType //
    //
    //	TypeSection::RoadType	currentRoadType_;
};

#endif // OPENSCENARIOEDITOR_HPP
