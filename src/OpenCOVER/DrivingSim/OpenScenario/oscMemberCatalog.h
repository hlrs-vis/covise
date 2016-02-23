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
class OPENSCENARIOEXPORT oscMemberCatalog: public oscMember, public unordered_map<std::string /*objectName*/, oscObjectBase *>
{
protected:
    std::string m_catalogType; ///< type of the objects in this catalog, e.g. vehicle, pedestrian
    unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> m_catalogTypeToTypeName; ///< typeName of the objects for catalogType
    unordered_map<std::string /*objectName*/, bf::path /*fileName*/> m_availableObjects; ///< objectName is the attribute name of the root element of file fileName

public:
    oscMemberCatalog(); ///< constructor
    virtual ~oscMemberCatalog(); ///< destructor

    //
    std::vector<bf::path> getXoscFilesFromDirectory(const bf::path &pathToDirectory); ///< find xosc file recursively in given directory
    void fastReadCatalogObjects(const std::vector<bf::path> &filenames); ///< parse files and add objectName and filePath to map m_availableObjects

    //catalogType
    void setCatalogType(const std::string &catalogType);
    std::string getCatalogType() const;

    //m_availableObjects
    void setAvailableObjectsMap(const unordered_map<std::string , bf::path> &availableObjects);
    unordered_map<std::string , bf::path> getAvailableObjectsMap() const;
    bool addObjToAvailableObjectsMap(const std::string &objectName, const bf::path &pathToFile);
    bool removeObjFromAvailableObjectsMap(const std::string &objectName);

    //m_objectsInMemory
    //read a file into object structure (with object name or with a fileName given)
    //add an object to map (objectName and oscObjectBase) (and to availableObjects)
    //get a pointer to object
    //remove an object from map (objectName and oscObjectBase) and availableObjects
    //
    bool fullReadCatalogObjectWithName(const std::string &objectName); ///< read file for given objectName, generate the object structure and add object to map m_objectsInMemory
};

}

#endif /* OSC_MEMBER_CATALOG_H */
