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
class oscPrivateAction;
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
	explicit OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *catalog, const QPointF &pos, const QString &roadId);
    virtual ~OSCItem();


    // Garbage //
    //
    virtual bool deleteRequest();

    // Graphics //
    //
	void updateColor(const std::string &type);
	
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

    bool removeElement();

    //################//
    // EVENTS         //
    //################//

public:
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
    QString updateName();

	OpenScenario::oscObject *oscObject_;
	OpenScenario::oscObjectBase *catalogObject_;
	OpenScenario::oscPrivateAction *oscPrivateAction_;
	OpenScenario::oscCatalog *catalog_;
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
