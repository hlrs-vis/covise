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
class OPENSCENARIOEXPORT oscMemberCatalog: public oscMember, public unordered_map<int /*object refId*/, oscObjectBase *>
{
public:
    typedef unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> CatalogTypeTypeNameMap;
    typedef unordered_map<int /*object refId*/, bf::path /*fileName*/> AvailableObjectsMap;
    typedef unordered_map<int /*object refId*/, oscObjectBase *> ObjectsInMemoryMap;

protected:
    static const CatalogTypeTypeNameMap s_catalogTypeToTypeName; ///< typeName of the objects for catalogType
    std::string m_catalogType; ///< type of the objects in this catalog, e.g. vehicle, pedestrian
    AvailableObjectsMap m_availableObjects; ///< objectName is the attribute name of the root element of file fileName

public:
    oscMemberCatalog(); ///< constructor
    virtual ~oscMemberCatalog(); ///< destructor

    //
    std::vector<bf::path> getXoscFilesFromDirectory(const bf::path &pathToDirectory); ///< find xosc file recursively in given directory
    void fastReadCatalogObjects(const std::vector<bf::path> &filenames); ///< parse files and add objectRefId and filePath to m_availableObjects

    //catalogType
    void setCatalogType(const std::string &catalogType);
    std::string getCatalogType() const;

    //m_availableObjects
    void setMapAvailableObjects(const AvailableObjectsMap &availableObjects);
    AvailableObjectsMap getMapAvailableObjects() const;
    bool addObjToMapAvailableObjects(const int objectRefId, const bf::path &fileNamePath);
    bool removeObjFromMapAvailableObjects(const int objectRefId);
    void deleteMapAvailableObject();

    //m_objectsInMemory
    bool fullReadCatalogObjectWithName(const int objectRefId); ///< read file for given objectRefId, generate the object structure and add object to oscMemberCatalog map
    bool fullReadCatalogObjectFromFile(const bf::path &fileNamePath); ///< read file, get objectRefId, check and add to m_availableObjects, generate the object structure and add object to map m_objectsInMemory
    bool addCatalogObject(oscObjectBase *objectBase); ///< read objectRefId and fileNamePath from oscObjectBase and add entries to m_availableObjects and oscMemberCatalog map
    bool addCatalogObject(const int objectRefId, oscObjectBase *objectBase, bf::path &fileNamePath); ///< add objectRefId and fileName to m_availableObjects, add objectRefId and objectPtr to oscMemberCatalog map
    bool removeCatalogObject(const int objectRefId); ///< remove object with refId objectRefId from oscMemberCatalog map
    oscObjectBase *getCatalogObject(const int objectRefId); ///< return pointer to oscObjectBase for objectRefId from oscMemberCatalog map
    void deleteMapObjectsInMemory();

private:
    typedef std::pair<bool, int> SuccessIntVar;

    SuccessIntVar getObjectRefIdFromFile(const bf::path &fileNamePath); ///< return refId of the catalog object in file fileNamePath
    SuccessIntVar getIntFromIntAttribute(xercesc::DOMAttr *attribute); ///< read an attribute of type oscMemberValue::INT and return int
};

}

#endif /* OSC_MEMBER_CATALOG_H */
