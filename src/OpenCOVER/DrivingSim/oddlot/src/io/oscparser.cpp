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
#include "oscObjectBase.h"
#include "oscFileHeader.h"


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
#include <QDebug>

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
OSCParser::OSCParser(OpenScenario::OpenScenarioBase *openScenarioBase, QObject *parent)
    : QObject(parent)
	, openScenarioBase_(openScenarioBase)
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
OSCParser::parseXOSC(const QString &filename, const QString &nodeName, const QString &fileType)
{

    // Mode //
    //
    mode_ = OSCParser::MODE_XOSC;

/*	oscFactories * factories = oscFactories::instance();
	oscFactory<oscObjectBase,std::string> factory;
	factories->setObjectFactory(&factory);
	factory.create(tr("Driver").toStdString());*/


	if (openScenarioBase_->loadFile(filename.toStdString(), nodeName.toStdString(), fileType.toStdString()) == false)
    {
        qDebug() << "failed to load OpenScenarioBase from file " << filename;
        return false;
    }
    // <OpenSCENARIO> //
    //
	// TODO: validation of files should be selectable
	//
	//enable/disable validation of parsed files of type fileType (OpenSCENARIO or catalog object files, e.g. vehicle, driver)
/*	bool validate = openScenarioBase_->getValidation();
	xercesc::DOMElement *root = openScenarioBase_->getRootElement(filename.toStdString(), nodeName.toStdString(), fileType.toStdString(), validate);

	QString tagName =  xercesc::XMLString::transcode(root->getTagName());

    if (tagName != "OpenSCENARIO")
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Root element is not <OpenSCENARIO>!"));
        return false;
    }

    // <OpenSCENARIO><header> //
    //
	const OpenScenario::oscObjectBase * h = openScenarioBase_->fileHeader.getObject();
    
    if (!h)
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Missing <header> element!"));
        return false;
    }
    else
    {
        parseHeaderElement(child);
    } */

//	createElements(dynamic_cast<OpenScenario::oscObjectBase *>(openScenarioBase_));

    return true;
}

void
OSCParser::createElements(const OpenScenario::oscObjectBase *object)
{
	OpenScenario::oscObjectBase::MemberMap members = object->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
    {
        oscMember *member = it->second;
        if(member)
        {
			if(member->getType() == oscMemberValue::OBJECT)
            {
				oscObjectBase *memberObject = member->getObject();
				if (memberObject)
				{
					OSCElement *oscElement = new OSCElement(QString::fromStdString(it->first), memberObject); 
					oscBase_->addOSCElement(oscElement);
					createElements(memberObject);
				}
			}
		}
	}
}




