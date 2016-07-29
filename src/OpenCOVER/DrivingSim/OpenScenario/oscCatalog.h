/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_BASE_H
#define OSC_CATALOG_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"

#include "oscDirectory.h"
#include "oscUserDataList.h"

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




namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalog: public oscObjectBase
{
	typedef struct 
	{
		bf::path fileName;
		oscObjectBase *object;
	} ObjectParams;

public:
    oscCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(directory, "oscDirectory");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscDirectoryMember directory;
    oscUserDataListArrayMember userDataList;

	
    typedef unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> CatalogTypeTypeNameMap;
	typedef unordered_map<int, ObjectParams> ObjectsMap; ///< represent the unordered_map of objects
	
protected:
    static const CatalogTypeTypeNameMap s_catalogTypeToTypeName; ///< typeName of the objects for catalogType
    std::string m_catalogType; ///< type of the objects in this catalog, e.g. vehicle, pedestrian
	ObjectsMap m_Objects;
	
public:
    //
    std::vector<bf::path> getXoscFilesFromDirectory(const bf::path &pathToDirectory); ///< find xosc file recursively in given directory
    void fastReadCatalogObjects(const std::vector<bf::path> &filenames); ///< parse files and add objectRefId and filePath to ObjectsMap

    //catalogType
    void setCatalogType(const std::string &catalogType);
    std::string getCatalogType() const;

    //availableObjects
    void setObjectsMap(const ObjectsMap &availableObjects);
    ObjectsMap getObjectsMap() const;
    bool addObjToObjectsMap(const int objectRefId, const bf::path &fileNamePath, oscObjectBase *object);
    bool removeObjFromObjectsMap(const int objectRefId);
	std::string getPath(const int objectRefId);

    //ObjectsInMemory
    bool fullReadCatalogObjectWithName(const int objectRefId); ///< read file for given objectRefId, generate the object structure and add object to ObjectsMap map
    bool fullReadCatalogObjectFromFile(const bf::path &fileNamePath); ///< read file, get objectRefId, check and add to ObjectsMap, generate the object structure and add object to ObjectsMap 
    bool addCatalogObject(oscObjectBase *objectBase); ///< read objectRefId and fileNamePath from oscObjectBase and add entries to ObjectsMap
    bool addCatalogObject(const int objectRefId, oscObjectBase *objectBase, const bf::path &fileNamePath); ///< add objectRefId and fileName and objectPtr to ObjectsMap
    bool removeCatalogObject(const int objectRefId); ///< remove object with refId objectRefId from ObjectsMap
    oscObjectBase *getCatalogObject(const int objectRefId); ///< return pointer to oscObjectBase for objectRefId from ObjectsMap

	//s_catalogTypeToTypeName
	std::string getType(const std::string &typeName);

	//generate refId for new object
	int generateRefId();

	// write all catalog members to catalogs
	void writeCatalogToDOM();
	void writeCatalogToDisk();
	
    virtual bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src);
    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document);

private:
    typedef std::pair<bool, int> SuccessIntVar;

    SuccessIntVar getObjectRefIdFromFile(const bf::path &fileNamePath); ///< return refId of the catalog object in file fileNamePath
    SuccessIntVar getIntFromIntAttribute(xercesc::DOMAttr *attribute); ///< read an attribute of type oscMemberValue::INT and return int
};

typedef oscObjectVariable<oscCatalog *> oscCatalogMember;

}

#endif /* OSC_CATALOG_BASE_H */
