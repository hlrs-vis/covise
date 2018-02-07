/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#include "graphscene.hpp"

//  //
//
#include "src/gui/mouseaction.hpp"
#include "src/gui/keyaction.hpp"

#include "src/mainwindow.hpp"

// Qt //
//
#include <QGraphicsSceneMouseEvent>
#include <QKeyEvent>

GraphScene::GraphScene(const QRectF &sceneRect, QObject *parent)
    : QGraphicsScene(sceneRect, parent)
    , Observer()
{
    init();
}

GraphScene::~GraphScene()
{
    //	// Observer //
    //	//
    //	dataElement_->detachObserver(this);
}

//################//
// FUNCTIONS      //
//################//

void
GraphScene::init()
{
    setBackgroundBrush(QBrush(QColor(238, 243, 238)));
	

    //	// Observer //
    //	//
    //	dataElement_->attachObserver(this);
}

void
GraphScene::mouseAction(MouseAction *mouseAction)
{
    if (mouseAction->getMouseActionType() == MouseAction::ATM_PRESS)
    {
        QGraphicsScene::mousePressEvent(mouseAction->getEvent());
    }
    else if (mouseAction->getMouseActionType() == MouseAction::ATM_MOVE)
    {
        QGraphicsScene::mouseMoveEvent(mouseAction->getEvent());
    }
    else if (mouseAction->getMouseActionType() == MouseAction::ATM_RELEASE)
    {
        QGraphicsScene::mouseReleaseEvent(mouseAction->getEvent());
    }
    else if (mouseAction->getMouseActionType() == MouseAction::ATM_DOUBLECLICK)
    {
        QGraphicsScene::mouseDoubleClickEvent(mouseAction->getEvent());
    }
}

void
GraphScene::keyAction(KeyAction *keyAction)
{
    if (keyAction->getKeyActionType() == KeyAction::ATK_PRESS)
    {
        QGraphicsScene::keyPressEvent(keyAction->getEvent());
    }
    else if (keyAction->getKeyActionType() == KeyAction::ATK_RELEASE)
    {
        QGraphicsScene::keyReleaseEvent(keyAction->getEvent());
    }
}

//################//
// SLOTS          //
//################//

//################//
// MOUSE EVENTS   //
//################//

void
GraphScene::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    // Forward Event to Selected Item //
    //
    //QGraphicsScene::mousePressEvent(mouseEvent);

    // Don't Forward Event to Selected Item //
    //
    // Problem: at this point it is not known if the event should be intercepted,
    // e.g. by the addTrackComponent tool.
    // Instead: Pass the event to the ProjectWidget and let the ProjectWidget
    // decide what to do. The event may eventually come back as an MouseAction (see above).

    MouseAction *mouseAction = new MouseAction(MouseAction::ATM_PRESS, mouseEvent);
    emit(mouseActionSignal(mouseAction));
    delete mouseAction;
}

void
GraphScene::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    // Status Bar //
    //
    ODD::mainWindow()->updateStatusBarPos(mouseEvent->scenePos());

    // Forward Event to Selected Item //
    //
    //QGraphicsScene::mouseMoveEvent(mouseEvent);

    MouseAction *mouseAction = new MouseAction(MouseAction::ATM_MOVE, mouseEvent);
    emit(mouseActionSignal(mouseAction));
    delete mouseAction;
}

void
GraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    // Forward Event to Selected Item //
    //
    //QGraphicsScene::mouseReleaseEvent(mouseEvent);

    MouseAction *mouseAction = new MouseAction(MouseAction::ATM_RELEASE, mouseEvent);
    emit(mouseActionSignal(mouseAction));
    delete mouseAction;
}

void
GraphScene::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent)
{
    // Forward Event to Selected Item //
    //
    //QGraphicsScene::mouseDoubleClickEvent(mouseEvent);

    MouseAction *mouseAction = new MouseAction(MouseAction::ATM_DOUBLECLICK, mouseEvent);
    emit(mouseActionSignal(mouseAction));
    delete mouseAction;
}

//################//
// DRAG EVENTS    //
//################//

void 
GraphScene::dropEvent(QGraphicsSceneDragDropEvent *event)
{
	MouseAction *mouseAction = new MouseAction(MouseAction::ATM_DROP, event);
	emit(mouseActionSignal(mouseAction));
	delete mouseAction; 
}

//################//
// KEY EVENTS     //
//################//

void
GraphScene::keyPressEvent(QKeyEvent *event)
{
    // Forward Event to Selected Item //
    //
    //QGraphicsScene::keyPressEvent(event);

    KeyAction *keyAction = new KeyAction(KeyAction::ATK_PRESS, event);
    emit(keyActionSignal(keyAction));
    delete keyAction;
}

void
GraphScene::keyReleaseEvent(QKeyEvent *event)
{
    // Forward Event to Selected Item //
    //
    //QGraphicsScene::keyReleaseEvent(event);

    KeyAction *keyAction = new KeyAction(KeyAction::ATK_RELEASE, event);
    emit(keyActionSignal(keyAction));
    delete keyAction;
}
