/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#ifndef OSCITEM_HPP
#define OSCITEM_HPP

#include "oscbaseitem.hpp"

namespace OpenScenario
{
class OpenScenarioBase;
class oscObjectBase;
class oscVehicle;
class oscPedestrian;
class oscObject;
class oscCatalog;
}

class RoadSystem;
class OpenScenarioEditor;
class OSCTextItem;
class OSCBaseItem;

class QColor;

class OSCItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *entityCatalog, const QPointF &pos, const QString &roadId);
    virtual ~OSCItem();


    // Garbage //
    //
    virtual bool deleteRequest();

    // Graphics //
    //
	void updateColor(const std::string &type);
//	void createVehiclePath();
//	void createPath(OpenScenario::oscPedestrian *pedestrian);
	
	// Function for path drawing //
	QPainterPath *(*createPath)(OpenScenario::oscObjectBase *);

    void updatePosition();


    // Garbage //
    //
    //	virtual void			notifyDeletion();

    // Obsever Pattern //
    //
    virtual void updateObserver();

	//################//
	// SIGNALS        //
	//################//

signals:
    void toolAction(ToolAction *);  // send action to copy the selected item //

    //################//
    // SLOTS          //
    //################//

public slots:

    //Tools
    //
 //   void zoomAction();

    bool removeElement();

    //################//
    // EVENTS         //
    //################//

public:
    //	virtual QVariant		itemChange(GraphicsItemChange change, const QVariant & value);
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	void keyReleaseEvent(QKeyEvent *event);

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
	OSCElement *element_;
	RoadSystem *roadSystem_;
	OSCBaseItem * oscBaseItem_;
	OSCRoadSystemItem *roadSystemItem_;
	QString roadID_;
    void init();

	OpenScenario::oscObject *oscObject_;
	OpenScenario::oscObjectBase *selectedObject_;
	OpenScenario::oscCatalog *entityCatalog_;
	RSystemElementRoad *road_;
	RSystemElementRoad *closestRoad_;

	double s_;
	double t_;
    QPointF pos_;
	QPainterPath *path_;

    QPointF pressPos_;
	QPointF lastPos_;
	bool doPan_;
	bool copyPan_;

    OSCTextItem *oscTextItem_;

    QColor color_;

    OpenScenarioEditor *oscEditor_;


};

#endif // OSCITEM_HPP
