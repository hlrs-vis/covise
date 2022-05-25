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
#ifndef SIGNALPOLEITEM_HPP
#define SIGNALPOLEITEM_HPP

#include <QGraphicsPixmapItem>
#include "src/data/observer.hpp"

class SignalSectionPolynomialItems;
class SignalManager;
class Signal;
class SignalEditor;

class SignalPoleItem : public QGraphicsPixmapItem, public Observer
{

public:    

    explicit SignalPoleItem(SignalSectionPolynomialItems *signalSectionPolynomialItems, Signal *signal, SignalEditor *signalEditor, SignalManager *signalManager);
    virtual ~SignalPoleItem();


    virtual void init();
    virtual void kill();
    virtual void updateObserver();


private:

    virtual void initPixmap();
    void transformPixmapIntoRightPresentation();
    void move(QPointF &diff);

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
    bool doPan_;

    //x = t | y = z
    QPointF pos_;
    QPointF pressPos_, lastPos_;
    Signal *signal_;
    SignalEditor *signalEditor_;
    SignalManager *signalManager_;
    SignalSectionPolynomialItems *signalSectionPolynomialItems_;
};

#endif // SIGNALPOLEITEM_HPP
