/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "oscbase.hpp"

// Data //
//
#include "oscelement.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/tilesystem/tile.hpp"

// OpenScenario data //
//
#include "oscObject.h"
#include "oscObjectBase.h"

using namespace OpenScenario;

/*! \brief The constructor does nothing special.
*
*/
OSCBase::OSCBase()
    : DataElement()       //   , rSystemElementChanges_(0x0)
	, openScenarioBase_(NULL)
{
}

OSCBase::OSCBase(OpenScenario::OpenScenarioBase *openScenarioBase)
	: DataElement()
	, openScenarioBase_(openScenarioBase)
{
}

OSCBase::~OSCBase()
{
}

//##################//
// ProjectData      //
//##################//

void
OSCBase::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
//    addScenerySystemChanges(ScenerySystem::CSC_ProjectDataChanged);
}


OSCElement *
OSCBase::getOSCElement(const QString &id) const
{
	return oscElements_.value(id);
}

OSCElement *
OSCBase::getOSCElement(OpenScenario::oscObjectBase *oscObjectBase)
{
	OSCElement *oscElement;

	QMap<QString, OSCElement *>::const_iterator it = oscElements_.constBegin();
	while (it != oscElements_.constEnd())
	{
		oscElement = it.value();
		if (oscObjectBase == oscElement->getObject())
		{
			return oscElement;
		}
		it++;
	}

	oscElement = new OSCElement(QString::fromStdString(oscObjectBase->getOwnMember()->getName()));
	oscElement->setObjectBase(oscObjectBase);
	addOSCElement(oscElement);

	return oscElement;
}

void 
OSCBase::addOSCElement(OSCElement *oscElement)
{
	if (getProjectData())
    {
        // Id //
        //
//		QString name = oscElement->getObject()->getName();
		QString name;
        QString id = getUniqueId(oscElement->getID(), name);
        if (id != oscElement->getID())
        {
            oscElement->setID(id);
        }
    }

    // Insert //
    //
    oscElement->setOSCBase(this);

    oscElements_.insert(oscElement->getID(), oscElement);
 //   addRoadSystemChanges(RoadSystem::CRS_RoadChange);
}

bool 
OSCBase::delOSCElement(OSCElement *oscElement)
{
	QStringList parts = oscElement->getID().split("_");

    if (oscElements_.remove(oscElement->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        oscElement->setOSCBase(NULL);
//		addOSCElementChanges(OSCBase::COSC_ElementChange);
        return true;
    }
    else
    {
        qDebug("WARNING 1005311350! Delete OpenScenario Element not successful!");
        return false;
    }

}

//##################//
// IDs              //
//##################//
const QString
OSCBase::getUniqueId(const QString &suggestion, QString &name)
{
    QString tileId = getProjectData()->getTileSystem()->getCurrentTile()->getID();
    QList<int> currentTileElementIds_ = elementIds_.values(tileId);

    // Try suggestion //
    //
    if (!suggestion.isNull() && !suggestion.isEmpty() && !name.isEmpty())
    {
        bool number = false;
        QStringList parts = suggestion.split("_");

        if (parts.size() > 2)
        {
            parts.at(0).toInt(&number);
            if (tileId == parts.at(0))
            {
                int nr = parts.at(1).toInt(&number);

                if (number && !currentTileElementIds_.contains(nr))
                {
                    elementIds_.insert(tileId, nr);
                    return suggestion;
                }
            }
        }
    }

    // Create new one //
    //

    if (name.isEmpty())
    {
        name = "unnamed";
    }
    /*	else if (name.contains("_"))       // get rid of old name concatention
	{
		int index = name.indexOf("_");
		name = name.left(index-1);
	}*/

    QString id;

    int index = 0;
    while ((index < currentTileElementIds_.size()) && currentTileElementIds_.contains(index))
    {
        index++;
    }

    id = QString("%1_%2_%3").arg(tileId).arg(index).arg(name);
    elementIds_.insert(tileId, index);
    return id;
}

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
OSCBase::accept(Visitor *visitor)
{
    visitor->visit(this);
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
OSCBase::notificationDone()
{
    oscElementChanges_ = 0x0;
    DataElement::notificationDone(); // pass to base class
}

/*! \brief Add one or more change flags.
*
*/
void
OSCBase::addOSCElementChanges(int changes)
{
    if (changes)
    {
        oscElementChanges_ |= changes;
        notifyObservers();
    }
}
