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

#ifndef OSCBASE_HPP
#define OSCBASE_HPP

#include "src/data/dataelement.hpp"


namespace OpenScenario
{
class oscObjectBase;
class oscObject;
}

class OSCElement;


class OSCBase : public DataElement
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
    explicit OSCBase(OpenScenario::oscObjectBase *oscObjectBase);
	explicit OSCBase();
    virtual ~OSCBase();

	void setOSCObjectBase(OpenScenario::oscObjectBase *oscObject)
	{
		oscObjectBase_ = oscObject;
	}

	OpenScenario::oscObjectBase *getOSCObjectBase()
	{
		return oscObjectBase_;
	}

	OSCElement *getOSCElement(const QString &id) const;
	OSCElement *getOSCElement(OpenScenario::oscObject *oscObject) const;

    QMap<QString, OSCElement *> getOSCElements() const
    {
        return oscElements_;
    }

    void addOSCElement(OSCElement *oscElement);
    bool delOSCElement(OSCElement *oscElement);

	const QString getUniqueId(const QString &suggestion, QString &name);

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
//    OSCBase(); /* not allowed */
    OSCBase(const OSCBase &); /* not allowed */
	OSCBase &operator=(const OSCBase &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
	// OpenScenario Base object //
	//
	OpenScenario::oscObjectBase *oscObjectBase_;

	// List of all OpenScenario objects
	//
	QMap<QString, OSCElement *> oscElements_;

	QMultiMap<QString, int> elementIds_;

    // Observer Pattern //
    //
//    int rSystemElementChanges_;
};

#endif // OSCBASE_HPP
