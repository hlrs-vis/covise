/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/25/2010
**
**************************************************************************/

#ifndef OSCTEXTITEM_HPP
#define OSCTEXTITEM_HPP

#include "src/graph/items/graphelement.hpp"
#include "src/graph/items/svgelement.hpp"


class OSCElement;
class OSCItem;
class TextHandle;

namespace OpenScenario
{
class oscObject;
}

class OSCTextItem : public GraphElement
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCTextItem(OSCElement *element, GraphElement *item, const QString &text, const QPointF &pos);
    virtual ~OSCTextItem();

    virtual void createPath();
    virtual QPainterPath shape() const;

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

    void updateText(const QString &text);

private:
    OSCTextItem(); /* not allowed */
    OSCTextItem(const OSCTextItem &); /* not allowed */
    OSCTextItem &operator=(const OSCTextItem &); /* not allowed */

    void updatePosition();

    //################//
    // SLOTS          //
    //################//

public slots:
    void handlePositionChange(const QPointF &pos);
    //	void						handleSelectionChange(bool selected);

    //################//
    // EVENTS         //
    //################//

protected:
    virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
    //	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
    //	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

    //	virtual void			hoverEnterEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverLeaveEvent(QGraphicsSceneHoverEvent * event);
    //	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

    //################//
    // PROPERTIES     //
    //################//

private:
	OSCElement *element_;
    QString text_;
    TextHandle *textHandle_;
	QPointF pos_;
};


class OSCTextSVGItem : public SVGElement
{
	Q_OBJECT

		//################//
		// FUNCTIONS      //
		//################//

public:
	explicit OSCTextSVGItem(OSCElement *element, SVGElement *item, const QString &text, const QPointF &pos);
	virtual ~OSCTextSVGItem();

	virtual void createPath();
	virtual QPainterPath shape() const;

	// Obsever Pattern //
	//
	virtual void updateObserver();

	// delete this item
	virtual bool deleteRequest()
	{
		return false;
	};

	void updateText(const QString &text);

private:
	OSCTextSVGItem(); /* not allowed */
	OSCTextSVGItem(const OSCTextSVGItem &); /* not allowed */
	OSCTextSVGItem &operator=(const OSCTextSVGItem &); /* not allowed */

	void updatePosition();

	//################//
	// SLOTS          //
	//################//

	public slots:
	void handlePositionChange(const QPointF &pos);
	//	void						handleSelectionChange(bool selected);

	//################//
	// EVENTS         //
	//################//

protected:
	virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	//	virtual void			mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
	//	virtual void			mouseMoveEvent(QGraphicsSceneMouseEvent * event);

	//	virtual void			hoverEnterEvent(QGraphicsSceneHoverEvent * event);
	//	virtual void			hoverLeaveEvent(QGraphicsSceneHoverEvent * event);
	//	virtual void			hoverMoveEvent(QGraphicsSceneHoverEvent * event);

	//################//
	// PROPERTIES     //
	//################//

private:
	OSCElement *element_;
	QString text_;
	TextHandle *textHandle_;

	QGraphicsPathItem *pathItem_;
	QPointF pos_;
};



#endif // OSCTextItem_HPP
