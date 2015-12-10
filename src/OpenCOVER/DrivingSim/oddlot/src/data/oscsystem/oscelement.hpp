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

#ifndef OSCELEMENT_HPP
#define OSCELEMENT_HPP

#include "src/data/dataelement.hpp"
#include "oscObject.h"

using namespace OpenScenario;


class OSCElement : public DataElement, public OpenScenario::oscObject
{

    //################//
    // STATIC         //
    //################//

public:
 /*   enum DRoadSystemElementType
    {
        DRE_None,
        DRE_Road,
        DRE_Controller,
        DRE_Junction,
        DRE_Fiddleyard,
        DRE_PedFiddleyard,
        DRE_Signal,
        DRE_Object
    };*/

/*    enum RSystemElementChange
    {
        CRE_IdChange = 0x1,
        CRE_NameChange = 0x2,
        CRE_ParentRoadSystemChange = 0x4
    };*/

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCElement(OpenScenario::oscObject *oscObject);
    virtual ~OSCElement();

	OpenScenario::oscObject *getObject()
	{
		return oscObject_;
	}

	// Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);


    // Observer Pattern //
    //
  /*  virtual void notificationDone();
    int getRSystemElementChanges() const
    {
        return rSystemElementChanges_;
    }
    void addRSystemElementChanges(int changes);*/

private:
    OSCElement(); /* not allowed */
    OSCElement(const OSCElement &); /* not allowed */
	OSCElement &operator=(const OSCElement &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
	OpenScenario::oscObject *oscObject_;

    // Observer Pattern //
    //
//    int rSystemElementChanges_;
};

#endif // OSCELEMENT_HPP
