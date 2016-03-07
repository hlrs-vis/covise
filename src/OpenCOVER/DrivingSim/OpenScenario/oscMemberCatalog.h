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
    static const unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> m_catalogTypeToTypeName; ///< typeName of the objects for catalogType
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
    void setMapAvailableObjects(const unordered_map<std::string , bf::path> &availableObjects);
    unordered_map<std::string , bf::path> getMapAvailableObjects() const;
    bool addObjToMapAvailableObjects(const std::string &objectName, const bf::path &fileNamePath);
    bool removeObjFromMapAvailableObjects(const std::string &objectName);
    void deleteMapAvailableObject();

    //m_objectsInMemory
    bool fullReadCatalogObjectWithName(const std::string &objectName); ///< read file for given objectName, generate the object structure and add object to map m_objectsInMemory
    bool fullReadCatalogObjectFromFile(const bf::path &fileNamePath); ///< read file, get objectName, check and add to m_availableObjects, generate the object structure and add object to map m_objectsInMemory
    bool addCatalogObject(oscObjectBase *objectBase); ///< read objectName and fileNamePath from oscObjectBase and add entries to m_availableObjects and oscMemberCatalog map
    bool addCatalogObject(const std::string &objectName, oscObjectBase *objectBase, bf::path &fileNamePath); ///< add objectName and fileName to m_availableObjects, add objectName and objectPtr to oscMemberCatalog map
    bool removeCatalogObject(const std::string &objectName); ///< remove object with name objectName from oscMemberCatalog map
    oscObjectBase *getCatalogObject(const std::string &objectName); ///< return pointer to oscObjectBase for objectName from oscMemberCatalog map
    void deleteMapObjectsInMemory();

private:
    std::string getObjectNameFromFile(const bf::path &fileNamePath); ///<return name of the catalog object in file fileNamePath
};

}

#endif /* OSC_MEMBER_CATALOG_H */
