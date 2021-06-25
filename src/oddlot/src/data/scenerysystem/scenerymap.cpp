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

#include "scenerymap.hpp"

#include "scenerysystem.hpp"

SceneryMap::SceneryMap(const QString &id, const QString &filename, double width, double height, SceneryMap::SceneryMapType mapType)
    : DataElement()
    , parentScenerySystem_(NULL)
    , sceneryMapChanges_(0x0)
    , id_(id)
    , filename_(filename)
    , x_(0.0)
    , y_(0.0)
    , width_(width)
    , height_(height)
    , opacity_(1.0)
    , imageWidth_(0)
    , imageHeight_(0)
    , loaded_(false)
    , scaling_(0)
    , mapType_(mapType)
{
    loadFile();
}

SceneryMap::~SceneryMap()
{
}

//###################//
// SceneryMap        //
//###################//

/*! \brief Set the map's x id.
*/
void
SceneryMap::setId(const QString &id)
{
    id_ = id;
    addSceneryMapChanges(SceneryMap::CSM_Id);
}

/*! \brief Set the map's x position.
*/
void
SceneryMap::setX(double x)
{
    x_ = x;
    addSceneryMapChanges(SceneryMap::CSM_X);
}

/*! \brief Set the map's y position.
*/
void
SceneryMap::setY(double y)
{
    y_ = y;
    addSceneryMapChanges(SceneryMap::CSM_Y);
}

/*! \brief Set the map's width.
*/
void
SceneryMap::setWidth(double width)
{
    if (width < 1.0)
    {
        width = 1.0;
    }

    width_ = width;
    addSceneryMapChanges(SceneryMap::CSM_Width);
}

/*! \brief Set the map's height.
*/
void
SceneryMap::setHeight(double height)
{
    if (height < 1.0)
    {
        height = 1.0;
    }

    height_ = height;
    addSceneryMapChanges(SceneryMap::CSM_Height);
}

/*! \brief Set .
*/
void
SceneryMap::setOpacity(double opacity)
{
    if (opacity < 0.0)
    {
        opacity = 0.0;
    }
    else if (opacity > 1.0)
    {
        opacity = 1.0;
    }

    opacity_ = opacity;
    addSceneryMapChanges(SceneryMap::CSM_Opacity);
}

void
SceneryMap::setFilename(const QString &filename)
{
    filename_ = filename;
    addSceneryMapChanges(SceneryMap::CSM_Filename);
    loadFile();
}

/*! \brief Returns the meters per pixel.
*/
double
SceneryMap::getScaling() const
{
    if (!scaling_)
    {
        scaling_ = getWidth() / getImageWidth();
    }
    return scaling_;
}

bool
SceneryMap::loadFile()
{
    QImage image_(filename_);
    if (image_.isNull())
    {
        qDebug("ERROR 1010081652! Image could not be loaded!");
        qDebug("%s", filename_.toUtf8().constData());
        loaded_ = false;
        return false;
    }
    else
    {
        loaded_ = true;
        imageWidth_ = image_.width();
        imageHeight_ = image_.height();
        return true;
    }
}

void
SceneryMap::setParentScenerySystem(ScenerySystem *scenerySystem)
{
    parentScenerySystem_ = scenerySystem;
    setParentElement(scenerySystem);
    addSceneryMapChanges(SceneryMap::CSM_ScenerySystemChanged);
}

/*! \brief Returns true if the point lies in the region of the map.
*
*/
bool
SceneryMap::isIntersectedBy(const QPointF &point)
{
    if ((point.x() > x_)
        && (point.x() < x_ + width_)
        && (point.y() > y_)
        && (point.y() < y_ + height_))
    {
        return true;
    }
    else
    {
        return false;
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
SceneryMap::notificationDone()
{
    sceneryMapChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
SceneryMap::addSceneryMapChanges(int changes)
{
    if (changes)
    {
        sceneryMapChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
SceneryMap::accept(Visitor *visitor)
{
    visitor->visit(this);
}
