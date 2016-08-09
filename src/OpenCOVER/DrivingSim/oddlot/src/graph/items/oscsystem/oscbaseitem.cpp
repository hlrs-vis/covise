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

#include "oscbaseitem.hpp"

// Data //
//
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/commands/osccommands.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphview.hpp"
#include "src/graph/graphscene.hpp"
//#include "src/graph/items/roadsystem/signal/signaltextitem.hpp"
#include "src/graph/items/oscsystem/oscitem.hpp"
#include "src/graph/items/roadsystem/scenario/oscroadsystemitem.hpp"
#include "src/graph/editors/osceditor.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"

// OpenScenario //
//
#include "oscVehicle.h"
#include "oscObject.h"
#include "oscObjectBase.h"
#include "oscMember.h"
#include "oscPosition.h"

// Qt //
//
#include <QBrush>
#include <QPen>
#include <QCursor>
#include <QColor>
#include <QString>
#include <QKeyEvent>

using namespace OpenScenario;

OSCBaseItem::OSCBaseItem(TopviewGraph *topviewGraph, OSCBase *oscBase)
	: GraphElement(NULL, oscBase)
	, topviewGraph_(topviewGraph)
	, oscBase_(oscBase)
	, oscRoadSystemItem_(NULL)
{
	roadSystem_ = topviewGraph->getProjectData()->getRoadSystem();
    init();
}

OSCBaseItem::~OSCBaseItem()
{
}

void
OSCBaseItem::init()
{
	OpenScenario::OpenScenarioBase *openScenarioBase = oscBase_->getOpenScenarioBase();
	OpenScenario::oscCatalog *entityCatalog = openScenarioBase->catalogs->getCatalog("entityCatalog");

	// Root Road item //
    //

	foreach (OSCElement *element, oscBase_->getOSCElements())
    {
		OpenScenario::oscObject *object = dynamic_cast<OpenScenario::oscObject *>(element->getObject());
		if (object)
		{
			OpenScenario::oscPosition *oscPosition = object->initPosition.getObject();

			OpenScenario::oscPositionRoad *oscPosRoad = oscPosition->positionRoad.getObject();
			QString roadId = QString::fromStdString(oscPosRoad->roadId.getValue());
			RSystemElementRoad *road = roadSystem_->getRoad(roadId);
			double s = oscPosRoad->s.getValue();
			double t = oscPosRoad->t.getValue();
			new OSCItem(this, object, entityCatalog, road->getGlobalPoint(s, t), roadId);
		}
    }

}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called when the observed DataElement has been changed.
*
*/
void
OSCBaseItem::updateObserver()
{
    // Parent //
    //
    GraphElement::updateObserver();
    if (isInGarbage())
    {
        return; // will be deleted anyway
    }

    // Signal //
    //
 /*   int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_TypeChange))
    {
        updatePosition();
    }
    else if ((changes & Signal::CEL_ParameterChange))
    {
        updatePosition();
    } */
}
