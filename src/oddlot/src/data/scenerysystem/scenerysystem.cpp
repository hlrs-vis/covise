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

#include "scenerysystem.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "scenerymap.hpp"
#include "heightmap.hpp"
#include "scenerytesselation.hpp"

ScenerySystem::ScenerySystem()
    : DataElement()
    , scenerySystemChanges_(0x0)
    , sceneryTesselation_(NULL)
    , idCount_(0)
{
}

ScenerySystem::~ScenerySystem()
{
    // Delete child nodes //
    //
    foreach (SceneryMap *child, sceneryMaps_)
        delete child;
}

const QString
ScenerySystem::getUniqueId(const QString &suggestion)
{
    // Try suggestion //
    //
    if (!suggestion.isNull())
    {
        if (!ids_.contains(suggestion))
        {
            ids_.append(suggestion);
            //			qDebug() << "Suggestion successful: " << suggestion;
            return suggestion;
        }
    }

    // Create new one //
    //
    QString id = QString("map%1").arg(idCount_);
    while (ids_.contains(id))
    {
        id = QString("map%1").arg(idCount_);
        ++idCount_;
    }
    //	qDebug() << "Suggestion not successful, used: " << id << ".";
    ++idCount_;
    ids_.append(id);
    return id;
}

//##################//
// BackgroundImages //
//##################//

void
ScenerySystem::addSceneryFile(const QString &filename)
{
    if (!sceneryFiles_.contains(filename))
    {
        sceneryFiles_.append(filename);
    }

    addScenerySystemChanges(ScenerySystem::CSC_FileChanged);
}

bool
ScenerySystem::delSceneryFile(const QString &filename)
{
    addScenerySystemChanges(ScenerySystem::CSC_FileChanged);
    return sceneryFiles_.removeOne(filename);
}

//##################//
// BackgroundImages //
//##################//

void
ScenerySystem::addSceneryMap(SceneryMap *map)
{
    QString id = getUniqueId(map->getId());
    map->setId(id);
    sceneryMaps_.insert(map->getId(), map);

    map->setParentScenerySystem(this);

    addScenerySystemChanges(ScenerySystem::CSC_MapChanged);
}

bool
ScenerySystem::delSceneryMap(SceneryMap *map)
{
    addScenerySystemChanges(ScenerySystem::CSC_MapChanged);

    if (sceneryMaps_.remove(map->getId()) && ids_.removeOne(map->getId()))
    {
        map->setParentScenerySystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1006111414! Delete map not successful!");
        return false;
    }
}

//##################//
// Heightmaps       //
//##################//

void
ScenerySystem::addHeightmap(Heightmap *map)
{
    QString id = getUniqueId(map->getId());
    map->setId(id);
    heightmaps_.insert(map->getId(), map);

    map->setParentScenerySystem(this);

    addScenerySystemChanges(ScenerySystem::CSC_HeightmapChanged);
}

bool
ScenerySystem::delHeightmap(Heightmap *map)
{
    addScenerySystemChanges(ScenerySystem::CSC_HeightmapChanged);

    if (heightmaps_.remove(map->getId()) && ids_.removeOne(map->getId()))
    {
        map->setParentScenerySystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1006111414! Delete map not successful!");
        return false;
    }
}

//##################//
// Heightmaps       //
//##################//

void
ScenerySystem::setSceneryTesselation(SceneryTesselation *sceneryTesselation)
{
    if (sceneryTesselation_)
    {
        qDebug("WARNING 1011181018! Tesselation Settings applied twice. Deleted first one.");
        sceneryTesselation_->setParentScenerySystem(NULL);
        delete sceneryTesselation_;
    }
    sceneryTesselation_ = sceneryTesselation;
    sceneryTesselation_->setParentScenerySystem(this);
    addScenerySystemChanges(ScenerySystem::CSC_TesselationChanged);
}

//##################//
// ProjectData      //
//##################//

void
ScenerySystem::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
    addScenerySystemChanges(ScenerySystem::CSC_ProjectDataChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
ScenerySystem::notificationDone()
{
    scenerySystemChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ScenerySystem::addScenerySystemChanges(int changes)
{
    if (changes)
    {
        scenerySystemChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
ScenerySystem::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
ScenerySystem::acceptForChildNodes(Visitor *visitor)
{
    acceptForMaps(visitor);
    acceptForHeightmaps(visitor);
    acceptForTesselation(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
ScenerySystem::acceptForMaps(Visitor *visitor)
{
    foreach (SceneryMap *child, sceneryMaps_)
    {
        child->accept(visitor);
    }
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
ScenerySystem::acceptForHeightmaps(Visitor *visitor)
{
    foreach (Heightmap *child, heightmaps_)
    {
        child->accept(visitor);
    }
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
ScenerySystem::acceptForTesselation(Visitor *visitor)
{
    if (sceneryTesselation_)
    {
        sceneryTesselation_->accept(visitor);
    }
}
