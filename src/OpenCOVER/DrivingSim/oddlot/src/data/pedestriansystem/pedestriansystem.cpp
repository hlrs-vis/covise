/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   01.06.2010
**
**************************************************************************/

#include "pedestriansystem.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "pedestriangroup.hpp"
#include <QList>

PedestrianSystem::PedestrianSystem()
    : DataElement()
    , pedestrianSystemChanges_(0x0)
    , parentProjectData_(NULL)
    , idCount_(0)
{
}

PedestrianSystem::~PedestrianSystem()
{
    // Delete child nodes //
    //
    foreach (PedestrianGroup *child, pedestrianGroups_)
    {
        delete child;
    }
}

//##################//
// PedestrianGroups //
//##################//

void
PedestrianSystem::addPedestrianGroup(PedestrianGroup *pedestrianGroup)
{
    pedestrianGroups_.append(pedestrianGroup);
    addPedestrianSystemChanges(PedestrianSystem::CVS_PedestrianGroupsChanged);

    pedestrianGroup->setParentPedestrianSystem(this);
}

//##################//
// IDs              //
//##################//

const QString
PedestrianSystem::getUniqueId(const QString &suggestion)
{
    // Try suggestion //
    //
    if (!suggestion.isNull())
    {
        if (!ids_.contains(suggestion))
        {
            ids_.append(suggestion);
            return suggestion;
        }
    }

    // Create new one //
    //
    QString id = QString("ped%1").arg(idCount_);
    while (ids_.contains(id))
    {
        id = QString("ped%1").arg(idCount_);
        ++idCount_;
    }
    ++idCount_;
    ids_.append(id);
    return id;
}

//##################//
// ProjectData      //
//##################//

void
PedestrianSystem::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
    addPedestrianSystemChanges(PedestrianSystem::CVS_ProjectDataChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
PedestrianSystem::notificationDone()
{
    pedestrianSystemChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
PedestrianSystem::addPedestrianSystemChanges(int changes)
{
    if (changes)
    {
        pedestrianSystemChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
*/
void
PedestrianSystem::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
PedestrianSystem::acceptForChildNodes(Visitor *visitor)
{
    acceptForPedestrianGroups(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
PedestrianSystem::acceptForPedestrianGroups(Visitor *visitor)
{
    foreach (PedestrianGroup *child, pedestrianGroups_)
    {
        child->accept(visitor);
    }
}
