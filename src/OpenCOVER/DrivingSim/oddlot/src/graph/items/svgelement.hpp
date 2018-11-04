/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public LicenseSVGElement
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#ifndef SVGElement_HPP
#define SVGElement_HPP


#include <QtSvg/QGraphicsSvgItem>
#include "src/graph/items/basegraphelement.hpp"


class SVGElement : public BaseGraphElement<QGraphicsSvgItem>
{
	Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SVGElement(SVGElement *parentElement, DataElement *dataElement);

    virtual ~SVGElement();

	//################//
	// SLOTS          //
	//################//

public slots:
	void hideGraphElement();
	virtual void hideRoads();

private:

 
};

#endif // SVGElement_HPP
