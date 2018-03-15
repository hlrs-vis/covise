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

#include "oscbaseshapeitem.hpp"

// Data //
//
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"

// Graph //
//
#include "src/graph/items/oscsystem/oscshapeitem.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// OpenScenario //
//
#include <OpenScenario/oscObjectBase.h>
#include <OpenScenario/oscMember.h>
#include <OpenScenario/schema/oscTrajectory.h>


// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

using namespace OpenScenario;

OSCBaseShapeItem::OSCBaseShapeItem(TopviewGraph *topviewGraph, OSCBase *oscBase)
	: GraphElement(NULL, oscBase)
	, oscBase_(oscBase)
	, topviewGraph_(topviewGraph)
{


    init();
}

OSCBaseShapeItem::~OSCBaseShapeItem()
{
}

void
OSCBaseShapeItem::init()
{	

	// Root Base item //
    //
	foreach(OSCElement *element, oscBase_->getOSCElements())
	{

		OpenScenario::oscTrajectory *trajectory = dynamic_cast<OpenScenario::oscTrajectory *>(element->getObject());
		if (trajectory)
		{
			new OSCShapeItem(element, this, trajectory);
		}
	}

}

void
OSCBaseShapeItem::appendOSCShapeItem(OSCShapeItem *oscShapeItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscShapeItem->getDataElement());
	QString id = element->getID();
    if (!oscShapeItems_.contains(id))
    {
        oscShapeItems_.insert(id, oscShapeItem);
    }
}

bool
OSCBaseShapeItem::removeOSCShapeItem(OSCShapeItem *oscShapeItem)
{
	OSCElement *element = dynamic_cast<OSCElement *>(oscShapeItem->getDataElement());
    return oscShapeItems_.remove(element->getID());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
OSCBaseShapeItem:: updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Base //
    //
	int changes = oscBase_->getOSCBaseChanges();

	if (changes & OSCBase::OSCBaseChange::COSC_ElementChange)
    {
		QMap<QString, OSCElement *>::const_iterator iter = oscBase_->getOSCElements().constBegin();
		while (iter != oscBase_->getOSCElements().constEnd())
		{
			OSCElement *element = iter.value();
			OpenScenario::oscTrajectory *trajectory = dynamic_cast<OpenScenario::oscTrajectory *>(element->getObject());
			if (trajectory)
			{
				if ((element->getDataElementChanges() & DataElement::CDE_DataElementCreated)
					|| (element->getDataElementChanges() & DataElement::CDE_DataElementAdded))
				{
					new OSCShapeItem(element, this, trajectory);
				}
			}
			iter++;
		}
	}

}
