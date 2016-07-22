/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MEMBER_CATALOG_H
#define OSC_MEMBER_CATALOG_H

#include "oscExport.h"
#include "oscMember.h"

#include <vector>
#if __cplusplus >= 201103L || defined WIN32
#include <unordered_map>
using std::unordered_map;
#else
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#endif

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


namespace bf = boost::filesystem;


namespace OpenScenario
{

class oscObjectBase;

/// \class This class represents a Member variable storing values of one kind of catalog
class OPENSCENARIOEXPORT oscCatalogMember: public oscMember, public unordered_map<int /*object refId*/, oscObjectBase *>
{
public:


    oscCatalogMember(); ///< constructor
    virtual ~oscCatalogMember(); ///< destructor

};

}

#endif /* OSC_MEMBER_CATALOG_H */
