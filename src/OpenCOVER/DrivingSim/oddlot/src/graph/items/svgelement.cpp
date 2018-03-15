/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#include "svgelement.hpp"

// Data //
//
#include "src/data/commands/dataelementcommands.hpp"


//################//
// CONSTRUCTOR    //
//################//


SVGElement::SVGElement(SVGElement *parentElement, DataElement *dataElement)
	: BaseGraphElement<QGraphicsSvgItem>(parentElement, dataElement)

{

}

SVGElement::~SVGElement()
{

}

//################//
// SLOTS          //
//################//

void
SVGElement::hideGraphElement()
{
	if (getDataElement())
	{
		QList<DataElement *> elements;
		elements.append(getDataElement());

		HideDataElementCommand *command = new HideDataElementCommand(elements, NULL);
		getProjectGraph()->executeCommand(command);
	} 
}

void
SVGElement::hideRoads()
{
	hideGraphElement();
}

