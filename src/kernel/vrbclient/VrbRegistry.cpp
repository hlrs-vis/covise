/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbRegistry.h"
#include "regClass.h"

regClass* vrb::VrbRegistry::getClass(const std::string& name) const
{

	auto cl = myClasses.find(name);
	if (cl == myClasses.end())
	{
		return nullptr;
	}
	return cl->second.get();
}