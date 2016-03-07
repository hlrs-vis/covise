/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscMemberCatalog.h"
#include "OpenScenarioBase.h"
#include "oscSourceFile.h"
#include "oscUtilities.h"

#include <iostream>

#include <boost/algorithm/string.hpp>

#include <xercesc/dom/DOMAttr.hpp>


namespace ba = boost::algorithm;


using namespace OpenScenario;


/*****
 * constructor
 *****/

oscMemberCatalog::oscMemberCatalog(): oscMember()
{

}


/*****
 * initialization static variables
 *****/

unordered_map<std::string /*m_catalogType*/, std::string /*catalogTypeName*/> initFuncCatToType()
{
    //set the typeName for possible catalogTypes
    unordered_map<std::string, std::string> catToType;
    catToType.emplace("driver", "oscDriver");
    catToType.emplace("entity", "oscEntity");
    catToType.emplace("environment", "oscEnvironment");
    catToType.emplace("maneuver", "oscManeuverTypeA");
    catToType.emplace("miscObject", "oscMiscObject");
    catToType.emplace("observer", "oscObserverTypeA");
    catToType.emplace("pedestrian", "oscPedestrian");
    catToType.emplace("routing", "oscRouting");
    catToType.emplace("vehicle", "oscVehicle");

    return catToType;
}

const unordered_map<std::string, std::string> oscMemberCatalog::m_catalogTypeToTypeName = initFuncCatToType();


/*****
 * destructor
 *****/

oscMemberCatalog::~oscMemberCatalog()
{

}



/*****
 * public functions
 *****/

std::vector<bf::path> oscMemberCatalog::getXoscFilesFromDirectory(const bf::path &pathToDirectory)
{
    //output vector
    std::vector<bf::path> fileNames;

    try
    {
        if (bf::exists(pathToDirectory))
        {
            if (bf::is_directory(pathToDirectory))
            {
                //bf::recursive_directory_iterator() constructs the end iterator
                for (bf::recursive_directory_iterator it(pathToDirectory); it != bf::recursive_directory_iterator(); it++)
                {
                    bf::path contentName = it->path();

                    if (bf::is_regular_file(contentName))
                    {
                        std::string lowerName = contentName.generic_string();
                        ba::to_lower(lowerName);
                        std::string extension = ".xosc";
                        std::size_t startPos = lowerName.size() - extension.size();

                        if (lowerName.compare(startPos, std::string::npos, extension) == 0)
                        {
                            fileNames.push_back(contentName);
                        }
                    }
                }
            }
            else
            {
                std::cerr << "Error! " << pathToDirectory << " is not a path to a directory." << std::endl;
            }
        }
        else
        {
            std::cerr << "Error! File or directory " << pathToDirectory << " do not exist." << std::endl;
        }
    }
    catch (const bf::filesystem_error& fse)
    {
        std::cerr << "getXoscFilesFromDirectory(): " << fse.what() << std::endl;
    }

    return fileNames;
}

void oscMemberCatalog::fastReadCatalogObjects(const std::vector<bf::path> &filenames)
{
    OpenScenarioBase *oscBase = new OpenScenarioBase();

    for (size_t i = 0; i < filenames.size(); i++)
    {
        xercesc::DOMElement *rootElem = oscBase->getRootElement(filenames[i].generic_string(), m_catalogType);

        if (rootElem)
        {
            std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

            if (rootElemName == m_catalogType)
            {
                xercesc::DOMAttr *attribute = rootElem->getAttributeNode(xercesc::XMLString::transcode("name"));
                if (attribute)
                {
                    std::string objectName = xercesc::XMLString::transcode(attribute->getValue());
                    unordered_map<std::string, bf::path>::const_iterator found = m_availableObjects.find(objectName);

                    if (objectName.empty())
                    {
                        std::cerr << "Error! Object for catalog " << m_catalogType << " in " << filenames[i] << " has an empty name and can't be used." << std::endl;
                    }
                    else if (found != m_availableObjects.end())
                    {
                        std::cerr << "Warning! Object for catalog " << m_catalogType << " with name " << objectName << " from " << filenames[i] << " is ignored." << std::endl;
                        std::cerr << "First appearance from file " << found->second << " is used." << std::endl;
                    }
                    else
                    {
                        m_availableObjects.emplace(objectName, filenames[i]);
                    }
                }
                else
                {
                    std::cerr << "Error! Can't find an object for catalog " << m_catalogType << " in file " << filenames[i] << " with attribute 'name'." << std::endl;
                }
            }
        }
    }

    delete oscBase;
}


//
void oscMemberCatalog::setCatalogType(const std::string &catalogType)
{
    m_catalogType = catalogType;
}

std::string oscMemberCatalog::getCatalogType() const
{
    return m_catalogType;
}


//
void oscMemberCatalog::setMapAvailableObjects(const unordered_map<std::string , bf::path> &availableObjects)
{
    m_availableObjects = availableObjects;
}

unordered_map<std::string , bf::path> oscMemberCatalog::getMapAvailableObjects() const
{
    return m_availableObjects;
}

bool oscMemberCatalog::addObjToMapAvailableObjects(const std::string &objectName, const bf::path &fileNamePath)
{
    if (objectName.empty())
    {
        std::cerr << "Error! Can't add the object with empty name for file " << fileNamePath << std::endl;
        return false;
    }

    unordered_map<std::string, bf::path>::const_iterator found = m_availableObjects.find(objectName);
    if (found != m_availableObjects.end())
    {
        std::cerr << "Error! Object with name " << objectName << " exists and is defined in file " << found->second << std::endl;
        return false;
    }

    std::pair<unordered_map<std::string, bf::path>::const_iterator, bool> returnVal = m_availableObjects.emplace(objectName, fileNamePath);
    if (returnVal.second == false)
    {
        std::cerr << "Error! Can't insert " << objectName << " from file " << fileNamePath << "into map of available objects." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

bool oscMemberCatalog::removeObjFromMapAvailableObjects(const std::string &objectName)
{
    if (objectName.empty())
    {
        std::cerr << "Error! Can't remove an object without a name." << std::endl;
        return false;
    }

    return m_availableObjects.erase(objectName);
}

void oscMemberCatalog::deleteMapAvailableObject()
{
    m_availableObjects.clear();
}


//
bool oscMemberCatalog::fullReadCatalogObjectWithName(const std::string &objectName)
{
    if (objectName.empty())
    {
        std::cerr << "Error! No object name specified." << std::endl;
        return false;
    }

    unordered_map<std::string, bf::path>::const_iterator found = m_availableObjects.find(objectName);
    if (found == m_availableObjects.end())
    {
        std::cerr << "Error! Object with name " << objectName << " isn't available. No file to read." << std::endl;
        return false;
    }

    bf::path filePath = found->second;
    if (bf::is_regular_file(filePath))
    {
        bool success = false;

        OpenScenarioBase *oscBase = new OpenScenarioBase;

        //in fullReadCatalogObjectWithName no validation should be done,
        //because during fastReadCatalogObjects validation is done
        xercesc::DOMElement *rootElem = oscBase->getRootElement(filePath.generic_string(), m_catalogType, false);
        if (rootElem)
        {
            std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

            if (rootElemName == m_catalogType)
            {
                unordered_map<std::string, std::string>::const_iterator found = m_catalogTypeToTypeName.find(m_catalogType);
                if (found != m_catalogTypeToTypeName.end())
                {
                    //sourceFile for objectName
                    oscSourceFile *srcFile = new oscSourceFile();

                    //set variables for srcFile, differentiate between absolute and relative path for catalog object
                    srcFile->setSrcFileHref(filePath);
                    srcFile->setSrcFileName(filePath.filename());
                    srcFile->setMainDocPath(owner->getSource()->getMainDocPath());
                    bf::path relPathFromMainDoc;
                    if (filePath.is_absolute())
                    {
                        relPathFromMainDoc = bf::path();
                    }
                    else
                    {
                        relPathFromMainDoc = owner->getSource()->getRelPathFromMainDoc();
                        relPathFromMainDoc /= filePath.parent_path();
                    }
                    srcFile->setRelPathFromMainDoc(relPathFromMainDoc);
                    srcFile->setRootElementName(rootElemName);

                    //object for objectName
                    std::string catalogTypeName = found->second;
                    oscObjectBase *obj = oscFactories::instance()->objectFactory->create(catalogTypeName);
                    if(obj)
                    {
                        obj->initialize(owner->getBase(), NULL, NULL, srcFile);
                        obj->parseFromXML(rootElem, srcFile);
                        //add objectName and object to map m_objectsInMemory
                        this->emplace(objectName, obj);
                        //add sourcFile to vector
                        owner->getBase()->addToSrcFileVec(srcFile);

                        success = true;
                    }
                    else
                    {
                        std::cerr << "Error! Could not create an object member of type " << catalogTypeName << std::endl;
                        delete srcFile;
                    }
                }
                else
                {
                    std::cerr << "Error! Can't determine an typeName for catalog " << m_catalogType << std::endl;
                }
            }
        }

        delete oscBase;

        return success;
    }
    else
    {
        std::cerr << "Error! Can't read from " << filePath << std::endl;
        return false;
    }
}

bool oscMemberCatalog::fullReadCatalogObjectFromFile(const bf::path &fileNamePath)
{
    bool success = false;

    std::string objectName = getObjectNameFromFile(fileNamePath);

    if (objectName != "")
    {
        if (addObjToMapAvailableObjects(objectName, fileNamePath))
        {
            if (fullReadCatalogObjectWithName(objectName))
            {
                success = true;
            }
        }
    }
    else
    {
        std::cerr << "Error: Object for catalog " << m_catalogType << " in " << fileNamePath << " has an empty name and can't be used." << std::endl;
    }

    return success;
}

bool oscMemberCatalog::addCatalogObject(oscObjectBase *objectBase)
{
    if (objectBase)
    {
        //get objectName
        //every object in a catalog (e.g. driver, vehicle ...) should have a member 'name' of type oscString
        std::string objectName;
        oscMember *objectNameMember = objectBase->getMembers()["name"];
        if (objectNameMember)
        {
            oscStringValue *objNameStringVal = dynamic_cast<oscStringValue *>(objectNameMember->getValue());
            objectName = objNameStringVal->getValue();
        }

        if (objectName.empty())
        {
            std::cerr << "Error: Can't determine name of the object." << std::endl;
            return false;
        }

        //get fileName and Path for file to write
        bf::path fileNamePath;
        oscSourceFile *objectSrc = objectBase->getSource();
        if (objectSrc)
        {
            fileNamePath = objectSrc->getSrcFileHref();
        }

        if (fileNamePath.empty())
        {
            std::cerr << "Error: Can't determine filename and path to write the object." << std::endl;
            return false;
        }

        return addCatalogObject(objectName, objectBase, fileNamePath);
    }
    else
    {
        std::cerr << "Error: Given pointer to object isn't accessible." << std::endl;
        return false;
    }
}

bool oscMemberCatalog::addCatalogObject(const std::string &objectName, oscObjectBase *objectBase, bf::path &fileNamePath)
{
    bool success = false;

    if (objectName != "" && objectBase != NULL && !fileNamePath.empty())
    {
        unordered_map<std::string, bf::path>::const_iterator foundAvailableObjects = m_availableObjects.find(objectName);
        unordered_map<std::string, oscObjectBase *>::const_iterator foundObjectsInMemory = this->find(objectName);

        if (foundAvailableObjects == m_availableObjects.end())
        {
            if (foundObjectsInMemory == this->end())
            {
                //add objectName and fileName to m_availableObjects
                if (addObjToMapAvailableObjects(objectName, fileNamePath))
                {
                    //add objectName and objectPtr to oscMemberCatalog map (objects in memory)
                    std::pair<unordered_map<std::string, oscObjectBase *>::const_iterator, bool> returnValObjInMem = this->emplace(objectName, objectBase);
                    if (returnValObjInMem.second == true)
                    {
                        success = true;
                    }
                    else
                    {
                        std::cerr << "Error! Can't insert object with name" << objectName << " to catalog " << m_catalogType << std::endl;
                    }
                }
            }
            else
            {
                std::cerr << "Error: Can't add catalog object "  << objectName << ". An object with this name is already registered." << std::endl;
            }
        }
        else
        {
            std::cerr << "Error: Can't add catalog object " << objectName << ". Object is read from file " << foundAvailableObjects->second << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: Can't add catalog object " << objectName << ". Empty objectName or filename or no pointer to object." << std::endl;
    }

    return success;
}

bool oscMemberCatalog::removeCatalogObject(const std::string &objectName)
{
    unordered_map<std::string, oscObjectBase *>::const_iterator found = this->find(objectName);
    if (found != this->end())
    {
        this->erase(found);
        return true;
    }
    else
    {
        std::cerr << "Error: Can't remove object with name " << objectName << " from catalog " << m_catalogType << ". Object not found." << std::endl;
        return false;
    }
}

oscObjectBase *oscMemberCatalog::getCatalogObject(const std::string &objectName)
{
    unordered_map<std::string, oscObjectBase *>::const_iterator found = this->find(objectName);
    if (found != this->end())
    {
        return found->second;
    }
    else
    {
        return NULL;
    }
}

void oscMemberCatalog::deleteMapObjectsInMemory()
{
    this->clear();
}



/*****
 * private functions
 *****/

std::string oscMemberCatalog::getObjectNameFromFile(const bf::path &fileNamePath)
{
    std::string objectName;
    OpenScenarioBase *oscBase = new OpenScenarioBase;
    xercesc::DOMElement *rootElem = oscBase->getRootElement(fileNamePath.generic_string(), m_catalogType);

    if (rootElem)
    {
        std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

        if (rootElemName == m_catalogType)
        {
            xercesc::DOMAttr *attribute = rootElem->getAttributeNode(xercesc::XMLString::transcode("name"));
            if (attribute)
            {
                objectName = xercesc::XMLString::transcode(attribute->getValue());
            }
        }
    }

    delete oscBase;

    return objectName;
}
