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
                std::cerr << pathToDirectory << " is not a path to a directory." << std::endl;
            }
        }
        else
        {
            std::cerr << "file or directory " << pathToDirectory << " do not exist." << std::endl;
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
    OpenScenarioBase *oscBase = new OpenScenarioBase;

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
            else
            {
                //std::cerr << "Error! File " << filenames[i] << " doesn't contain an object for catalog " << m_catalogType << std::endl;
            }
        }
        else
        {
            //std::cerr << "Error! Can't parse file "<< filenames[i] << std::endl;
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
void oscMemberCatalog::setAvailableObjectsMap(const unordered_map<std::string , bf::path> &availableObjects)
{
    m_availableObjects = availableObjects;
}

unordered_map<std::string , bf::path> oscMemberCatalog::getAvailableObjectsMap() const
{
    return m_availableObjects;
}

bool oscMemberCatalog::addObjToAvailableObjectsMap(const std::string &objectName, const bf::path &pathToFile)
{
    if (objectName.empty())
    {
        std::cerr << "Error! Can't add the object with empty name for file " << pathToFile << std::endl;
        return false;
    }

    unordered_map<std::string, bf::path>::const_iterator found = m_availableObjects.find(objectName);
    if (found != m_availableObjects.end())
    {
        std::cerr << "Error! Object with name " << objectName << " exists and is defined in file " << found->second << std::endl;
        return false;
    }

    std::pair<unordered_map<std::string, bf::path>::const_iterator, bool> returnVal = m_availableObjects.emplace(objectName, pathToFile);
    if (returnVal.second != true)
    {
        std::cerr << "Error! Can't insert " << objectName << " from file " << pathToFile << "into map of available objects" << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}

bool oscMemberCatalog::removeObjFromAvailableObjectsMap(const std::string &objectName)
{
    if (objectName.empty())
    {
        std::cerr << "Error! Can't remove an object without a name." << std::endl;
        return false;
    }

    return m_availableObjects.erase(objectName);
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
        OpenScenarioBase *oscBase = new OpenScenarioBase;
        bool success;

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
                    //
                    //if filePath is absolute, it must be checked during write: TODO!
                    //
                    srcFile->setSrcFileHref(filePath);
                    srcFile->setSrcFileName(filePath.filename());
                    srcFile->setMainDocPath(owner->getSource()->getMainDocPath());
                    //if relative paths are used
                    bf::path relPathFromMainDoc = owner->getSource()->getRelPathFromMainDoc();
                    relPathFromMainDoc /= filePath.parent_path();
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
                        success = false;
                    }
                }
                else
                {
                    std::cerr << "Error! Can't determine an typeName for catalog " << m_catalogType << std::endl;
                    success = false;
                }
            }
            else
            {
                //std::cerr << "Error! File " << filePath << " doesn't contain an object for catalog " << m_catalogType << std::endl;
                success = false;
            }
        }
        else
        {
            //std::cerr << "Error! Can't parse file "<< filePath << std::endl;
            success = false;
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
