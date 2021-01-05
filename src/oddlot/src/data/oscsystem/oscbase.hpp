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
#include "src/data/roadsystem/odrID.hpp"


namespace OpenScenario
{
class oscObjectBase;
class OpenScenarioBase;
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

    enum OSCBaseChange
    {
        COSC_ElementChange = 0x1
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCBase(OpenScenario::OpenScenarioBase *openScenarioBase);
	explicit OSCBase();
    virtual ~OSCBase();

	// ProjectData //
	//
	void setParentProjectData(ProjectData *projectData);

	void setOpenScenarioBase(OpenScenario::OpenScenarioBase *openScenarioBase)
	{
		openScenarioBase_ = openScenarioBase;
	}

	OpenScenario::OpenScenarioBase *getOpenScenarioBase()
	{
		return openScenarioBase_;
	}

	OSCElement *getOSCElement(OpenScenario::oscObjectBase *oscObjectBase);
	OSCElement *getOSCElement(const QString &id) const;
	OSCElement *getOrCreateOSCElement(OpenScenario::oscObjectBase *oscObjectBase);

    QMap<QString, OSCElement *> getOSCElements() const
    {
        return oscElements_;
    }

    void addOSCElement(OSCElement *oscElement);
    bool delOSCElement(OSCElement *oscElement);

	const QString getUniqueId(const QString &suggestion, const QString &name);

	// Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getOSCBaseChanges() const
    {
        return oscBaseChanges_;
    } 
    void addOSCBaseChanges(int changes);

private:
//    OSCBase(); /* not allowed */
    OSCBase(const OSCBase &); /* not allowed */
	OSCBase &operator=(const OSCBase &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
	// ProjectData //
	//
	ProjectData *parentProjectData_;

	// OpenScenario Base object //
	//
	OpenScenario::OpenScenarioBase *openScenarioBase_;

	// List of all OpenScenario objects
	//
	QMap<QString, OSCElement *> oscElements_;


    // Observer Pattern //
    //
    int oscBaseChanges_;
};

#endif // OSCBASE_HPP
