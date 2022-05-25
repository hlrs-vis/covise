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
#include "signalpoleitem.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/signalmanager.hpp"

// Graph //
//
#include "src/graph/editors/signaleditor.hpp"


// Signal //
//
#include "src/graph/items/roadsystem/signal/signalsectionpolynomialitems.hpp"


#include <QUndoStack>

///
/// \brief Constructor.
/// \param signal
/// \param signalEditor
/// \param signalManager
/// \param signalSectionPolynomialItems
/// 
SignalPoleItem::SignalPoleItem(SignalSectionPolynomialItems *signalSectionPolynomialItems, Signal *signal, SignalEditor *signalEditor, SignalManager *signalManager)
    : QGraphicsPixmapItem(signalSectionPolynomialItems)
    , Observer()
    , signal_(signal)
    , signalEditor_(signalEditor)
    , signalManager_(signalManager)
{
    init();
}

///
/// \brief Destructor.
///
SignalPoleItem::~SignalPoleItem()
{
    kill();
}

///
/// \brief Deletes in class created pointer.
///
void
SignalPoleItem::kill()
{
    signal_->detachObserver(this);
}

///
/// \brief Initialize variables coor, profilegraph and signalManager.
///
void
SignalPoleItem::init()
{
 //   signalPoleSystemItem_->addSignalPoleItem(signal_, this);

    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(ItemIsFocusable);
    signal_->attachObserver(this);

    pos_ = QPointF(signal_->getT(), signal_->getZOffset());
    initPixmap();
}

///
/// \brief Initialize pixmap and pixmapItem.
///
void
SignalPoleItem::initPixmap()
{
    SignalContainer *signalContainer = signalManager_->getSignalContainer(signal_->getCountry(), signal_->getType(), signal_->getTypeSubclass(), signal_->getSubtype());
    if(signalContainer)
    {
        QIcon icon = signalContainer->getSignalIcon();
        if (icon.availableSizes().count() > 0)
        {
            setPixmap(icon.pixmap(icon.availableSizes().first()));
            transformPixmapIntoRightPresentation();
        }
    }
}

///
/// \brief Transform and rotate pixmap into right presentiation.
///
void
SignalPoleItem::transformPixmapIntoRightPresentation()
{
    // Icons mirrored => transform and rotate
    QTransform trafo;
    double scale = 0.01;
    double width;
    double height;
    double x;
    QPointF pos = pos_;

    width = pixmap().width()*scale;
    height = scale * pixmap().height();
    x = pos.x() - width/2;
    trafo.translate(x, pos.y() + height/2);

    trafo.scale(scale, scale);
    trafo.rotate(180, Qt::XAxis);

    setTransform(trafo);
}


///
/// \brief
/// \param diff
///
void
SignalPoleItem::move(QPointF &diff)
{
    pos_ += diff;
    transformPixmapIntoRightPresentation();
}

///
/// \brief SignalPoleItem::hoverEnterEvent
/// \param event
///
void
SignalPoleItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{

    setCursor(Qt::OpenHandCursor);
    setFocus();

    // Parent //
    //
    QGraphicsPixmapItem::hoverEnterEvent(event); // pass to baseclass
}

///
/// \brief SignalPoleItem::hoverLeaveEvent
/// \param event
///
void
SignalPoleItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    QGraphicsPixmapItem::hoverLeaveEvent(event); // pass to baseclass
}

///
/// \brief SignalPoleItem::hoverMoveEvent
/// \param event
///
void
SignalPoleItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    // Parent //
    //
    QGraphicsPixmapItem::hoverMoveEvent(event);
}

///
/// \brief
/// \param event
///
void
SignalPoleItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    lastPos_ = pressPos_ = event->scenePos();
    doPan_ = true;

    QGraphicsPixmapItem::mousePressEvent(event);
}

///
/// \brief SignalPoleItem::mouseMoveEvent
/// \param event
///
void
SignalPoleItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if(doPan_)
    {
        QPointF newPos = event->scenePos();
        QPointF diff = QPointF(0, newPos.y() - lastPos_.y());
        move(diff);

       
        lastPos_ = newPos;
    }

    QGraphicsPixmapItem::mouseMoveEvent(event);
}

///
/// \brief SignalPoleItem::mouseReleaseEvent
/// \param event
///
void
SignalPoleItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{

    if (doPan_)
    {
        QPointF diff = QPointF(0, lastPos_.y() - pressPos_.y());

        Signal::SignalProperties props = signal_->getProperties();
        props.zOffset = signal_->getZOffset() + diff.y();
        SetSignalPropertiesCommand *command = new SetSignalPropertiesCommand(signal_, signal_->getId(), signal_->getName(), props, signal_->getValidity(), signal_->getSignalUserData());
        signalEditor_->getProjectGraph()->executeCommand(command);

        doPan_ = false;
    }

    QGraphicsPixmapItem::mouseReleaseEvent(event);
}

void
SignalPoleItem::updateObserver()
{
    // Signal //
    //
    int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_TypeChange))
    {
        initPixmap();
    }
}
