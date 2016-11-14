/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOGS_H
#define OSC_CATALOGS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCatalog.h"



namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogs: public oscObjectBase
{
public:
    oscCatalogs()
    {
        OSC_OBJECT_ADD_MEMBER(VehicleCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(DriverCatalog, "oscCatalog");       
        OSC_OBJECT_ADD_MEMBER(PedestrianCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(PedestrianControllerCatalog, "oscCatalog");
		OSC_OBJECT_ADD_MEMBER(MiscObjectCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(EnvironmentCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(ManeuverCatalog, "oscCatalog");
		OSC_OBJECT_ADD_MEMBER(TrajectoryCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(RouteCatalog, "oscCatalog");
        
    };

    oscCatalogMember VehicleCatalog;
    oscCatalogMember DriverCatalog;
    oscCatalogMember PedestrianCatalog;
	oscCatalogMember PedestrianControllerCatalog;
    oscCatalogMember MiscObjectCatalog;
    oscCatalogMember EntityCatalog;
    oscCatalogMember EnvironmentCatalog;
	oscCatalogMember ManeuverCatalog;
    oscCatalogMember TrajectoryCatalog;
    oscCatalogMember RouteCatalog;
   

	oscCatalog *getCatalog(const std::string &s);
};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;

}

#endif //OSC_CATALOGS_H
