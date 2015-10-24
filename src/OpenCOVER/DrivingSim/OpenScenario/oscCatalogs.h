/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_CATALOGS_H
#define OSC_CATALOGS_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

#include <oscVehicleCatalog.h>
#include <oscDriverCatalog.h>
#include <oscObserverCatalog.h>
#include <oscPedestrianCatalog.h>
#include <oscMiscObjectCatalog.h>
#include <oscEntityCatalog.h>
#include <oscEnvironmentCatalog.h>
#include <oscManeuverCatalog.h>
#include <oscRoutingCatalog.h>

#include <oscDirectory.h>
#include <oscUserData.h>
#include <oscFile.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogs: public oscObjectBase
{
public:
    oscCatalogs()
    {
		OSC_ADD_MEMBER(vehicle);
		OSC_ADD_MEMBER(driver);
		OSC_ADD_MEMBER(observer);
		OSC_ADD_MEMBER(pedestrian);
		OSC_ADD_MEMBER(miscObject);
		OSC_ADD_MEMBER(entity);
		OSC_ADD_MEMBER(environment);
		OSC_ADD_MEMBER(maneuver);
		OSC_ADD_MEMBER(routing);
		OSC_ADD_MEMBER(userData);
		OSC_ADD_MEMBER(include);
		
    };
	oscVehicleCatalogMember vehicle;
	oscDriverCatalogMember driver;
	oscObserverCatalogMember observer;
	oscPedestrianCatalogMember pedestrian;
	oscMiscObjectCatalogMember miscObject;
	oscEntityCatalogMember entity;
	oscEntityCatalogMember environment;
	oscManeuverCatalogMember maneuver;
	oscRoutingCatalogMember routing;
	oscUserDataMember userData;
	oscFileMember include;
    
};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;

}

#endif //OSC_CATALOGS_H