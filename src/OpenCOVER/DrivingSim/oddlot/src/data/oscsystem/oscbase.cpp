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
#include <OpenScenario/schema/oscObject.h>
#include <OpenScenario/oscObjectBase.h>

using namespace OpenScenario;

/*! \brief The constructor does nothing special.
*
*/
OSCBase::OSCBase()
    : DataElement()       
	, openScenarioBase_(NULL)
	, oscBaseChanges_(0x0)
{
}

OSCBase::OSCBase(OpenScenario::OpenScenarioBase *openScenarioBase)
	: DataElement()
	, openScenarioBase_(openScenarioBase)
	, oscBaseChanges_(0x0)
{
}

OSCBase::~OSCBase()
{
	oscElements_.clear();

	openScenarioBase_ = NULL;
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

	return NULL;
}


OSCElement *
OSCBase::getOSCElement(const QString &id) const
{
	return oscElements_.value(id);
}

OSCElement *
OSCBase::getOrCreateOSCElement(OpenScenario::oscObjectBase *oscObjectBase)
{
	OSCElement *oscElement = getOSCElement(oscObjectBase);

	if (!oscElement)
	{
		oscElement = new OSCElement("element");
		oscElement->setObjectBase(oscObjectBase);
		addOSCElement(oscElement);
	}

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
	oscElement->notifyParent();

	addOSCBaseChanges(OSCBaseChange::COSC_ElementChange);
}

bool 
OSCBase::delOSCElement(OSCElement *oscElement)
{
	QStringList parts = oscElement->getID().split("_");

	bool number = false;
	int tn = parts.at(1).toInt(&number);
	odrID tid;
	tid.setID(tn);
	getProjectData()->getTileSystem()->getTile(tid)->removeOSCID(oscElement->getID());
    if (oscElements_.remove(oscElement->getID()))
    {
        oscElement->setOSCBase(NULL);

		addOSCBaseChanges(OSCBase::COSC_ElementChange);
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
OSCBase::getUniqueId(const QString &suggestion, const QString &name)
{
	// oscIDs should be unique within a tile so ask the tile for a new ID
    return  getProjectData()->getTileSystem()->getCurrentTile()->getUniqueOSCID(suggestion,name);

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
    oscBaseChanges_ = 0x0;
    DataElement::notificationDone(); // pass to base class
}

/*! \brief Add one or more change flags.
*
*/
void
OSCBase::addOSCBaseChanges(int changes)
{
    if (changes)
    {
        oscBaseChanges_ |= changes;
        notifyObservers();
    }
}
