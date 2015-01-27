/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   6/11/2010
**
**************************************************************************/

#include "scenerymapitem.hpp"

// Data //
//
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/commands/scenerycommands.hpp"

// Graph //
//
#include "src/graph/items/scenerysystem/scenerysystemitem.hpp"

SceneryMapItem::SceneryMapItem(ScenerySystemItem *parentScenerySystem, SceneryMap *sceneryMap)
    : GraphElement(parentScenerySystem, sceneryMap)
    , sceneryMap_(sceneryMap)
    , pixmapItem_(NULL)
    , loaded_(false)
    , locked_(false)
{
    // Load Image //
    //
    loadFile();

    // Flags //
    //
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);

    // Transformation //
    //
    // Note: The y-Axis must be flipped so the image is not mirrored
    QTransform trafo;
    trafo.rotate(180, Qt::XAxis);
    setTransform(trafo);

    // Selection/Highlighting //
    //
    setSelectable();
    setOpacitySettings(1.0, 1.0);
    enableHighlighting(false); // do not use automatic highlighting (selection/hovering)

    // ContextMenu //
    //
    QAction *hideAction = getHideMenu()->addAction(tr("Map"));
    connect(hideAction, SIGNAL(triggered()), this, SLOT(hideGraphElement()));

    QAction *removeAction = getRemoveMenu()->addAction(tr("Map"));
    connect(removeAction, SIGNAL(triggered()), this, SLOT(removeMap()));

    // Parameters //
    //
    updateOpacity();
    updatePosition();
    updateSize();
}

SceneryMapItem::~SceneryMapItem()
{
    delete pixmapItem_;
}

bool
SceneryMapItem::loadFile()
{
    QPixmap pixmap(sceneryMap_->getFilename());
    if (pixmap.isNull())
    {
        qDebug("ERROR 1006111429! Pixmap could not be loaded!");
        loaded_ = false;
        return false;
    }
    else
    {
        // Pixmap //
        //
        delete pixmapItem_;
        pixmapItem_ = new QGraphicsPixmapItem(pixmap);
        pixmapItem_->setParentItem(this);

        // Path //
        //
        QPainterPath path; // A path for selecting and moving (hidden behind the pixmap item)
        path.addRect(0.0, 0.0, pixmap.width(), pixmap.height());
        setPath(path);

        loaded_ = true;
        return true;
    }
}

void
SceneryMapItem::updateSize()
{
    if (sceneryMap_->isLoaded())
    {
        double widthScale = sceneryMap_->getWidth() / pixmapItem_->pixmap().width();
        double heightScale = sceneryMap_->getHeight() / pixmapItem_->pixmap().height();

        QTransform trafo;
        trafo.rotate(180, Qt::XAxis);
        trafo.scale(widthScale, heightScale);
        setTransform(trafo);
    }
}

void
SceneryMapItem::updatePosition()
{
    setPos(sceneryMap_->getX(), sceneryMap_->getY() + sceneryMap_->getHeight()); // left bottom corner
}

void
SceneryMapItem::updateOpacity()
{
    setOpacity(sceneryMap_->getOpacity());
}

void
SceneryMapItem::updateFilename()
{
    loadFile();
}

void
SceneryMapItem::setLocked(bool locked)
{
    locked_ = locked;
    setFlag(QGraphicsItem::ItemIsMovable, !locked);
    setFlag(QGraphicsItem::ItemIsSelectable, !locked);
}

void
SceneryMapItem::setMapX(double x)
{
    SetMapPositionCommand *command = new SetMapPositionCommand(sceneryMap_, x, sceneryMap_->getY());
    getProjectGraph()->executeCommand(command);
}

void
SceneryMapItem::setMapY(double y)
{
    SetMapPositionCommand *command = new SetMapPositionCommand(sceneryMap_, sceneryMap_->getX(), y);
    getProjectGraph()->executeCommand(command);
}

void
SceneryMapItem::setMapWidth(double width, bool keepRatio)
{
    // Height //
    //
    double height = sceneryMap_->getHeight();
    if (keepRatio)
    {
        height = pixmapItem_->pixmap().height() * width / pixmapItem_->pixmap().width();
    }

    // Command //
    //
    SetMapSizeCommand *command = new SetMapSizeCommand(sceneryMap_, width, height);
    getProjectGraph()->executeCommand(command);
}

void
SceneryMapItem::setMapHeight(double height, bool keepRatio)
{
    // Width //
    //
    double width = sceneryMap_->getHeight();
    if (keepRatio)
    {
        width = pixmapItem_->pixmap().width() * height / pixmapItem_->pixmap().height();
    }

    // Command //
    //
    SetMapSizeCommand *command = new SetMapSizeCommand(sceneryMap_, width, height);
    getProjectGraph()->executeCommand(command);
}

//################//
// SLOTS          //
//################//

bool
SceneryMapItem::removeMap()
{
    DelMapCommand *command = new DelMapCommand(sceneryMap_);
    return getProjectGraph()->executeCommand(command);
}

//################//
// EVENTS         //
//################//

/*!
* Handles Item Changes.
*/
QVariant
SceneryMapItem::itemChange(GraphicsItemChange change, const QVariant &value)
{

    if (change == QGraphicsItem::ItemPositionChange)
    {
        QPointF newPos = value.toPointF();

        SetMapPositionCommand *command = new SetMapPositionCommand(sceneryMap_, newPos.x(), newPos.y() - sceneryMap_->getHeight());
        getProjectGraph()->executeCommand(command);

        return QPointF(sceneryMap_->getX(), sceneryMap_->getY() + sceneryMap_->getHeight()); // do not move
    }

    return GraphElement::itemChange(change, value);
}

//##################//
// Observer Pattern //
//##################//

void
SceneryMapItem::updateObserver()
{

    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // ignore other changes
    }

    // Get change flags //
    //
    int changes = sceneryMap_->getSceneryMapChanges();

    // SceneryMap //
    //
    if ((changes & SceneryMap::CSM_X)
        || (changes & SceneryMap::CSM_Y))
    {
        updatePosition();
    }

    if ((changes & SceneryMap::CSM_Width)
        || (changes & SceneryMap::CSM_Height))
    {
        updateSize();
    }

    if (changes & SceneryMap::CSM_Opacity)
    {
        updateOpacity();
    }

    if (changes & SceneryMap::CSM_Filename)
    {
        updateFilename();
    }

    //	if(changes & SceneryMap::CSM_Id)
    //	{
    //	}
}

//*************//
// Delete Item
//*************//

bool
SceneryMapItem::deleteRequest()
{
    return false;
}
