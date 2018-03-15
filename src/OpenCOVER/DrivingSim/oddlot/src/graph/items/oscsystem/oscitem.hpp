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

#include "osctextitem.hpp"
#include "src/graph/items/svgelement.hpp"

#include <QtSvg/QGraphicsSvgItem>

namespace OpenScenario
{
class OpenScenarioBase;
class oscObjectBase;
class oscVehicle;
class oscPedestrian;
class oscObject;
class oscCatalog;
class oscPrivateAction;
class oscPosition;
class oscRoad;
}

class OSCRoadSystemItem;
class RoadSystem;
class OpenScenarioEditor;

class OSCBaseItem;
class OSCTextItem;

class QColor;

class OSCItem : public SVGElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCItem(OSCElement *element, OSCBaseItem *oscBaseItem, OpenScenario::oscObject *oscObject, OpenScenario::oscCatalog *catalog, OpenScenario::oscRoad *oscRoad);
    virtual ~OSCItem();


    // Garbage //
    //
    virtual bool deleteRequest();

	OpenScenario::oscObject *getObject()
	{
		return oscObject_;
	}


    void updatePosition();
	void move(QPointF &diff);


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

protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);

    //################//
    // PROPERTIES     //
    //################//

private:
	OSCElement *element_;
	OSCElement *cloneElement_;
	OpenScenario::oscObject *oscObject_;
	OpenScenario::oscRoad *oscRoad_;
	RoadSystem *roadSystem_;
	OSCBaseItem * oscBaseItem_;
	OSCRoadSystemItem *roadSystemItem_;
	std::string covisedir_;
	

    void init();
	void updateIcon(OpenScenario::oscObjectBase *catalogObject, std::string catalogName, std::string categoryName, std::string entryName);
    QString updateName();

	OpenScenario::oscCatalog *catalog_;
	RSystemElementRoad *road_;
	RSystemElementRoad *closestRoad_;

	double s_;
	double t_;
	double angle_;
	double iconScaleX_;
	double iconScaleY_;
	QPointF svgCenter_;
    QPointF pos_;
	QPointF lastPos_;
	QPainterPath path_;

    QPointF mousePressPos_;
	QPointF mouseLastPos_;
	bool doPan_;
	bool copyPan_;

	QGraphicsSvgItem *cloneSvgItem_;
	std::string fn_;
	QTransform tR_;
	QTransform tS_;
	QTransform tT_;
	

	OSCTextSVGItem *oscTextItem_;

    QColor color_;

    OpenScenarioEditor *oscEditor_;

	QSvgRenderer *renderer_;


};

#endif // OSCITEM_HPP
