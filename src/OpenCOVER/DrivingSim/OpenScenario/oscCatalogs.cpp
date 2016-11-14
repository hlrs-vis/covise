/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.oscCatalog

* License: LGPL 2+ */

#include "oscCatalogs.h"
#include "OpenScenarioBase.h"



using namespace OpenScenario;



/*****
 * public functions
 *****/

oscCatalog *oscCatalogs::getCatalog(const std::string &s)
{
	oscMember *m = members[s];

	if (m)
	{
		oscObjectBase *obj = m->getOrCreateObject();
		if (obj)
		{
			return dynamic_cast<oscCatalog *>(obj);
		}
	}

	return NULL;
}