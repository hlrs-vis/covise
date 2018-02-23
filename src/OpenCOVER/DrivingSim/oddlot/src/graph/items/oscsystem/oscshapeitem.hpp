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

#ifndef OSCSHAPEITEM_HPP
#define OSCSHAPEITEM_HPP

#include "oscbaseshapeitem.hpp"
#include "src/graph/items/graphelement.hpp"

namespace OpenScenario
{
class OpenScenarioBase;
class oscObjectBase;
class oscTrajectory;
}

class OpenScenarioEditor;

class OSCTextItem;

class OSCBaseItem;

class QColor;

class OSCShapeItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCShapeItem(OSCElement *element, OSCBaseShapeItem *oscBaseShapeItem, OpenScenario::oscTrajectory *trajectory);
    virtual ~OSCShapeItem();


    // Garbage //
    //
    virtual bool deleteRequest();

	// Function for path drawing //
	virtual void createPath();

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
    OpenScenario::oscTrajectory *trajectory_;
	OSCElement *element_;
	OSCBaseShapeItem * oscBaseShapeItem_;
    void init();
    void createControlPoints();
    QString updateName();

	OpenScenario::oscObjectBase *selectedObject_;

    QPointF pos_;
	QPainterPath *path_;

    QPointF pressPos_;
	QPointF lastPos_;
	bool doPan_;
	bool copyPan_;

    OSCTextItem *oscTextItem_;

    QColor color_;

    OpenScenarioEditor *oscEditor_;

    QVector<QPointF> controlPoints_;


};

#endif // OSCSHAPEITEM_HPP
