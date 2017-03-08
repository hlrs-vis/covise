/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_BASE_H
#define OSC_CATALOG_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"

#include "schema/oscDirectory.h"
#include "schema/oscUserDataList.h"

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
	
public:
	typedef struct 
	{
		bf::path fileName;
		oscObjectBase *object;
	} ObjectParams;

    oscCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(Directory, "oscDirectory", 0);
    };

    oscDirectoryMember Directory;

	
    typedef unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> CatalogTypeTypeNameMap;
	typedef unordered_map<std::string, ObjectParams> ObjectsMap; ///< represent the unordered_map of objects
	
protected:
    std::string m_catalogName; ///< type of the objects in this catalog, e.g. vehicle, pedestrian
	std::string m_catalogType;
	ObjectsMap m_Objects;
	
public:
    //
    std::vector<bf::path> getXoscFilesFromDirectory(const bf::path &pathToDirectory); ///< find xosc file recursively in given directory
    void fastReadCatalogObjects(const std::vector<bf::path> &filenames); ///< parse files and add objectRefId and filePath to ObjectsMap

    //catalogType
    void setCatalogNameAndType(const std::string &catalogName);
    std::string getCatalogName() const;
	std::string getCatalogType() const;

    //availableObjects
    void setObjectsMap(const ObjectsMap &availableObjects);
    ObjectsMap getObjectsMap() const;
    bool addObjToObjectsMap(const std::string &name, const bf::path &fileNamePath, oscObjectBase *object);
    bool removeObjFromObjectsMap(const std::string &name);
	std::string getPath(const std::string &name);
	std::string getObjectPath(OpenScenario::oscObjectBase *object);
	OpenScenario::oscObjectBase *getObjectfromPath(const std::string &path);

    //ObjectsInMemory
    bool fullReadCatalogObjectWithName(const std::string &name); ///< read file for given objectRefId, generate the object structure and add object to ObjectsMap map
    bool fullReadCatalogObjectFromFile(const bf::path &fileNamePath); ///< read file, get objectRefId, check and add to ObjectsMap, generate the object structure and add object to ObjectsMap 
    bool addCatalogObject(oscObjectBase *objectBase); ///< read objectRefId and fileNamePath from oscObjectBase and add entries to ObjectsMap
    bool addCatalogObject(const std::string &name, oscObjectBase *objectBase, const bf::path &fileNamePath); ///< add objectRefId and fileName and objectPtr to ObjectsMap
    bool removeCatalogObject(const std::string &name); ///< remove object with refId objectRefId from ObjectsMap
    oscObjectBase *getCatalogObject(const std::string &name); ///< return pointer to oscObjectBase for objectRefId from ObjectsMap


	//generate refId for new object
	std::string generateRefId(int startId);

	// write all catalog members to catalogs
	void writeCatalogToDOM();
	void clearDOM();
	void writeCatalogToDisk();
	
    virtual bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src, bool saveInclude = true);
    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude = true);

private:
    typedef std::pair<bool, int> SuccessIntVar;

    std::string getObjectNameFromFile(const bf::path &fileNamePath); ///< return refId of the catalog object in file fileNamePath
    SuccessIntVar getIntFromIntAttribute(xercesc::DOMAttr *attribute); ///< read an attribute of type oscMemberValue::INT and return int
};

typedef oscObjectVariable<oscCatalog *> oscCatalogMember;

}

#endif /* OSC_CATALOG_BASE_H */
