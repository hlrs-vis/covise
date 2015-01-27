/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/12/2010
**
**************************************************************************/

#include "heightmap.hpp"

#include "scenerysystem.hpp"

#include <QFile>

#include "math.h"

Heightmap::Heightmap(const QString &id, const QString &filename, double width, double height, SceneryMap::SceneryMapType mapType, const QString &heightMapFilename)
    : SceneryMap(id, filename, width, height, mapType)
    , heightmapFilename_(heightMapFilename)
    , heightmapData_(NULL)
    , heightmapDataLoaded_(false)
{
    loadHeightmapDataFile();
}

Heightmap::~Heightmap()
{
}

bool
Heightmap::loadHeightmapDataFile()
{
    heightmapData_ = NULL;
    heightmapDataArray_.clear();

    addHeightmapChanges(Heightmap::CHM_DataFileChanged);

    // Read file //
    //
    QFile file(heightmapFilename_);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug("ERROR 1010121154! File could not be loaded!");
        qDebug(heightmapFilename_.toUtf8());
        heightmapDataLoaded_ = false;
        return false;
    }
    else
    {
        heightmapDataLoaded_ = true;

        // Save data //
        //
        heightmapDataArray_ = file.readAll();
        heightmapData_ = (double *)heightmapDataArray_.data();
        return true;
    }
}

double
Heightmap::getHeightmapValue(int i, int h) const
{
    if (heightmapDataLoaded_)
    {
        int newH = (getImageHeight() - 1) - h;
        int pixelPos = newH * getImageWidth() + i;
        return heightmapData_[pixelPos];
    }
    else
    {
        return 0;
    }
}

double
Heightmap::getHeightmapValue(double x, double y) const
{
    if (heightmapDataLoaded_)
    {
        int i = int(int(((x - getX()) / (getWidth() / (getImageWidth() - 1))) + 0.5));
        int h = int(int(((y - getY()) / (getHeight() / (getImageHeight() - 1))) + 0.5));

        return getHeightmapValue(i, h);
    }
    else
    {
        return 0;
    }
}

/*! \brief Sets the heightmap data filename and loads the file.
*
*/
void
Heightmap::setHeightmapDataFilename(const QString &filename)
{
    heightmapFilename_ = filename;
    loadHeightmapDataFile();
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Heightmap::notificationDone()
{
    heightmapChanges_ = 0x0;
    SceneryMap::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Heightmap::addHeightmapChanges(int changes)
{
    if (changes)
    {
        heightmapChanges_ |= changes;
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
Heightmap::accept(Visitor *visitor)
{
    visitor->visit(this);
}
