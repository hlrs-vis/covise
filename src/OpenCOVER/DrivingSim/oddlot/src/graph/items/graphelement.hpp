/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public LicenseGraphElement
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#ifndef GraphElement_HPP
#define GraphElement_HPP


#include <QGraphicsPathItem>
#include "src/graph/items/basegraphelement.hpp"

class GraphElement : public QObject, public BaseGraphElement<QGraphicsPathItem>
{
	Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit GraphElement(GraphElement *parentGraphElement, DataElement *dataElement);
    virtual ~GraphElement();


	//################//
	// SLOTS          //
	//################//

public slots:
	void hideGraphElement();
	virtual void hideRoads();

};


#endif // GraphElement_HPP
