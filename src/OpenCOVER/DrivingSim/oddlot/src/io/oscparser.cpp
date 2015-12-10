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

#include "oscparser.hpp"

//#include "src/mainwindow.hpp"

// Data Model //
//
#include "src/data/projectdata.hpp"
#include "src/data/oscsystem/oscbase.hpp"
#include "src/data/oscsystem/oscelement.hpp"

//#include "src/data/changemanager.hpp"

// OpenSCENARIO //
//
#include "oscFactories.h"
#include "oscFactory.h"
#include "OpenScenarioBase.h"
#include "oscObject.h"


/*#include "src/data/vehiclesystem/vehiclesystem.hpp"
#include "src/data/vehiclesystem/vehiclegroup.hpp"
#include "src/data/vehiclesystem/roadvehicle.hpp"
#include "src/data/vehiclesystem/poolvehicle.hpp"
#include "src/data/vehiclesystem/pool.hpp"
#include "src/data/vehiclesystem/carpool.hpp"

#include "src/data/pedestriansystem/pedestriansystem.hpp"
#include "src/data/pedestriansystem/pedestriangroup.hpp"
#include "src/data/pedestriansystem/pedestrian.hpp"*/


// Qt //
//
/*#include <QtGui>
#include <QDomDocument>*/
#include <QMessageBox>
#include <qDebug>

// Xercecs //
//
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"

using namespace OpenScenario;

/** CONSTRUCTOR.
*
*/
OSCParser::OSCParser(OpenScenario::OpenScenarioBase *base, QObject *parent)
    : QObject(parent)
    , base_(base)
    , mode_(MODE_NONE)
{
 //   doc_ = new QDomDocument();
	projectData_ = dynamic_cast<ProjectData *>(parent);
	oscBase_ = projectData_->getOSCBase();
}

/** DESTRUCTOR.
*
*/
OSCParser::~OSCParser()
{
 //   delete doc_;
}

//################//
// XODR           //
//################//

/*! \brief Opens a .xodr file, creates a DOM tree and reads in the first level.
*
*/
bool
OSCParser::parseXOSC(const QString &filename)
{

    // Mode //
    //
    mode_ = OSCParser::MODE_XOSC;

/*	oscFactories * factories = oscFactories::instance();
	oscFactory<oscObjectBase,std::string> factory;
	factories->setObjectFactory(&factory);
	factory.create(tr("Driver").toStdString());*/


	if(base_->loadFile(filename.toStdString())== false)
    {
        qDebug() << "failed to load OpenScenarioBase from file " << filename;
        return false;
    }
    // <OpenSCENARIO> //
    //
	xercesc::DOMElement * root = base_->getRootElement(filename.toStdString());

	QString tagName =  xercesc::XMLString::transcode(root->getTagName());

    if (tagName != "OpenScenario")
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Root element is not <OpenSCENARIO>!"));
        return false;
    }

    // <OpenSCENARIO><header> //
    //
	oscHeader * h = base_->header.getObject();
    
    if (!h)
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Missing <header> element!"));
        return false;
    }
    else
    {
 //       parseHeaderElement(child);
    }

	createElements(dynamic_cast<OpenScenario::oscObject *>(base_));

    return true;
}

void
OSCParser::createElements(OpenScenario::oscObject *object)
{
/*	OpenScenario::oscObjectBase::MemberMap members = object->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
    {
        oscMember *member = it->second;
        if(member)
        {
			if(member->getType() == oscMemberValue::OBJECT)
            {
				oscObject *memberObject = member->getValue(); // OSCMemberValue should be object
				OSCElement *oscElement = new OSCElement("prototype", memberObject); 
				oscBase_->addOSCElement(oscElement);
				createElements(memberObject);
			}
		}
	}*/
}




