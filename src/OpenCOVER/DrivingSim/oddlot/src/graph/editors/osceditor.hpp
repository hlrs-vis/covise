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
class oscCatalog;
class oscTrajectory;
class oscPrivateAction;
class oscArrayMember;
}

class OSCBaseItem;
class OSCBaseShapeItem;
class OSCItem;
class OSCRoadSystemItem;
class OSCElement;
class OSCBase;

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

    // get object list //
    //
    QList<OpenScenario::oscObjectBase *> getElements(OpenScenario::oscObjectBase *root, const std::string &type);

    // Move Object //
    //
	void move(QPointF &diff);
	void translate(QPointF &diff);
	void translateObject(OpenScenario::oscObject *oscObject, QPointF &diff);

	OpenScenario::oscCatalog *getCatalog(std::string name);
	OpenScenario::oscPrivateAction *getOrCreatePrivateAction(const std::string &selectedObjectName);
	std::string getName(OpenScenario::oscArrayMember *arrayMember, const std::string &baseName);
	OSCElement* cloneEntity(OSCElement *element, OpenScenario::oscObject *oscObject);


	// Catalog dock widget changed //
	//
	void catalogChanged(OpenScenario::oscCatalog *member);

	// Edit Trajectory element //
	//
	void setTrajectoryElement(OSCElement *trajectory);

	// New Object with properties chosen in SignalTreeWidget //
	//
//	Object *addObjectToRoad(RSystemElementRoad *road, double s, double t);
	OSCRoadSystemItem *getRoadSystemItem()
	{
		return oscRoadSystemItem_;
	}

    void addGraphToObserver(const QVector<QPointF> &controlPoints);
    void createWaypoints(OpenScenario::oscTrajectory *trajectory, const QVector<QPointF> &controlPoints);

	void enableSplineEditing(bool state);


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
	void changeDirectories();

    //################//
    // PROPERTIES     //
    //################//

private:

    // Handle //
    //
  //  OSCHandle *insertOSCHandle_;

	TopviewGraph *topviewGraph_;

	// RoadSystem //
	//
	 OSCRoadSystemItem * oscRoadSystemItem_;

	// MainWindow //
	//
	MainWindow * mainWindow_;

	// OpenScenarioBase //
	//
	OSCBase * oscBase_;
	OSCBaseItem * oscBaseItem_;
	OSCBaseShapeItem *oscBaseShapeItem_;
	OpenScenario::OpenScenarioBase *openScenarioBase_;

	// Selected catalog //
	//
	OpenScenario::oscCatalog *oscCatalog_;


	ODD::ToolId lastTool_;
	QString lastOSCObjectName_;

	QString catalogElement_;

	// Window with catalog entries //
	//
	CatalogTreeWidget *catalogTree_;


    // Selected waypoints //
    //
    OSCElement *trajectoryElement_;

    // RoadType //
    //
    //	TypeSection::RoadType	currentRoadType_;
};

#endif // OPENSCENARIOEDITOR_HPP
