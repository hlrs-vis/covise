/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_NAME_MAPPING_H
#define OSC_NAME_MAPPING_H

#include <string>
#include <utility>
#include <boost/bimap.hpp>

#include "oscExport.h"

namespace OpenScenario {

/// \class This is used to map OpenScenario schema names to class names
class OPENSCENARIOEXPORT nameMapping
{
	

public:
	typedef std::pair<std::string, std::string> parent_name;
	typedef boost::bimap< parent_name, std::string > bm_type;
	bm_type bm;
	typedef boost::bimap<std::string, std::string> eMap;
	eMap enumMap;
	nameMapping();
	static nameMapping *nmInstance;
	static nameMapping *instance();
	std::string getClassName(const std::string &name, std::string parent);
	std::string getSchemaName(std::string &className);
	std::string getEnumName(const std::string &name);
	std::string getSchemaEnumName(std::string &name);


	
};

}

#endif /* OSC_NAME_MAPPING_H */
