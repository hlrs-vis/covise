/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.oscCatalog

* License: LGPL 2+ */

#include "oscCatalog.h"
#include "OpenScenarioBase.h"
#include "oscSourceFile.h"
#include "oscUtilities.h"

#include <iostream>

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>

#include <boost/algorithm/string.hpp>


namespace ba = boost::algorithm;


using namespace OpenScenario;



/*****
 * initialization static variables
 *****/

oscCatalog::CatalogTypeTypeNameMap initFuncCatToType()
{
    //set the typeName for possible catalogTypes
    oscCatalog::CatalogTypeTypeNameMap catToType;
	catToType.emplace("Vehicle", "oscOpenSCENARIO_VehicleCatalog");
    catToType.emplace("Driver", "oscOpenSCENARIO_DriverCatalog");
	catToType.emplace("Pedestrian", "oscOpenSCENARIO_PedestrianCatalog");
    catToType.emplace("PedestrianController", "oscOpenSCENARIO_PedestrianControllerCatalog");
	catToType.emplace("MiscObject", "oscOpenSCENARIO_MiscObjectCatalog");
    catToType.emplace("Environment", "oscOpenSCENARIO_EnvironmentCatalog");
    catToType.emplace("Maneuver", "oscOpenSCENARIO_ManeuverCatalog");   
    catToType.emplace("Trajectory", "oscOpenSCENARIO_TrajectoryCatalog"); 
    catToType.emplace("Route", "oscOpenSCENARIO_RouteCatalog");
    

    return catToType;
}

const oscCatalog::CatalogTypeTypeNameMap oscCatalog::s_catalogNameToTypeName = initFuncCatToType();



/*****
 * public functions
 *****/

std::vector<bf::path> oscCatalog::getXoscFilesFromDirectory(const bf::path &pathToDirectory)
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
                std::cerr << "Warning! " << pathToDirectory << " is not a path to a directory." << std::endl;
            }
        }
        else
        {
            std::cerr << "Warning! File or directory " << pathToDirectory << " do not exist." << std::endl;
        }
    }
    catch (const bf::filesystem_error& fse)
    {
        std::cerr << "getXoscFilesFromDirectory(): " << fse.what() << std::endl;
    }

    return fileNames;
}

void oscCatalog::fastReadCatalogObjects(const std::vector<bf::path> &filenames)
{
    OpenScenarioBase *oscBase = new OpenScenarioBase();
    bool validate = getBase()->getValidation();

    for (size_t i = 0; i < filenames.size(); i++)
    {
        xercesc::DOMElement *rootElem = oscBase->getRootElement(filenames[i].generic_string(), m_catalogName, getType(m_catalogName), validate);

        if (rootElem)
        {
            std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

			/*           if (rootElemName == m_catalogName)
			{ */
			xercesc::DOMNodeList *list = rootElem->getElementsByTagName(xercesc::XMLString::transcode(m_catalogName.c_str()));

			xercesc::DOMNode *node = list->item(0);
			if (node)
			{
				xercesc::DOMNamedNodeMap *attributes = node->getAttributes();
				xercesc::DOMNode *attribute = attributes->getNamedItem(xercesc::XMLString::transcode("name"));


				//              xercesc::DOMAttr *attribute = node->getAttributeNode(xercesc::XMLString::transcode("name"));
				if (attribute)
				{
					/*                  SuccessIntVar successIntVar = getIntFromIntAttribute(attribute);
					if (successIntVar.first)
					{
					int objectRefId = successIntVar.second; */
					std::string attributeName = xercesc::XMLString::transcode(attribute->getNodeValue());
					ObjectsMap::const_iterator found = m_Objects.find(attributeName);

					if (found != m_Objects.end())
					{
						std::cerr << "Warning! Object for catalog " << m_catalogName << " with name " << attributeName << " from " << filenames[i] << " is ignored." << std::endl;
						std::cerr << "First appearance from file " << found->second.fileName << " is used." << std::endl;
					}
					else
					{
						ObjectParams param = { filenames[i], NULL};
						m_Objects.emplace(attributeName, param);
					}
					/*                  }
					else
					{
					std::cerr << "Warning! Object for catalog " << m_catalogName << " in " << filenames[i] << " has an invalid name and can't be used." << std::endl;
					} */
				}
				else
				{
					std::cerr << "Warning! Can't find an object for catalog " << m_catalogName << " in file " << filenames[i] << " with attribute 'attributeName'." << std::endl;
				}
			}
        }
    }

    delete oscBase;
}


//
void oscCatalog::setCatalogName(const std::string &catalogName)
{
    m_catalogName = catalogName;
}

std::string oscCatalog::getCatalogName() const
{
    return m_catalogName;
}


//
void oscCatalog::setObjectsMap(const ObjectsMap &availableObjects)
{
    m_Objects = availableObjects;
}

oscCatalog::ObjectsMap oscCatalog::getObjectsMap() const
{
    return m_Objects;
}

bool oscCatalog::addObjToObjectsMap(const std::string &name, const bf::path &fileNamePath, oscObjectBase *object)
{
    ObjectsMap::const_iterator found = m_Objects.find(name);
    if (found != m_Objects.end())
    {
		std::cerr << "Error! Object with name " << name << " exists and is defined in file " << found->second.fileName.string() << std::endl;
        return false;
    }

	ObjectParams params = { fileNamePath, object};
    std::pair<ObjectsMap::const_iterator, bool> returnVal = m_Objects.emplace(name, params);
    if (returnVal.second == false)
    {
        std::cerr << "Error! Can't insert name " << name << " from file " << fileNamePath << "into map of available objects." << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}
/// returns true if an element has been removed
bool oscCatalog::removeObjFromObjectsMap(const std::string &name)
{
	return (m_Objects.erase(name)>0); 
}

std::string oscCatalog::getType(const std::string &typeName)
{
	std::unordered_map<std::string, std::string>::const_iterator it = s_catalogNameToTypeName.find(typeName);
	return it->second;
}

//
bool oscCatalog::fullReadCatalogObjectWithName(const std::string &name)
{
    ObjectsMap::iterator objectFound = m_Objects.find(name);
    if (objectFound == m_Objects.end())
    {
        std::cerr << "Error! Object with name " << name << " isn't available. No file to read." << std::endl;
        return false;
    }

	bf::path filePath = objectFound->second.fileName;
    if (bf::is_regular_file(filePath))
    {
        bool success = false;

        OpenScenarioBase *oscBase = new OpenScenarioBase;

        //in fullReadCatalogObjectWithName no validation should be done,
        //because during fastReadCatalogObjects validation is done
        xercesc::DOMElement *rootElem = oscBase->getRootElement(filePath.generic_string(), m_catalogName, getType(m_catalogName), false);
        if (rootElem)
        {
            std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

 /*           if (rootElemName == m_catalogName)
            { */
                CatalogTypeTypeNameMap::const_iterator found = s_catalogNameToTypeName.find(m_catalogName);
                if (found != s_catalogNameToTypeName.end())
                {
                    //sourceFile for objectName
                    oscSourceFile *srcFile = new oscSourceFile();

                    //set variables for srcFile, differentiate between absolute and relative path for catalog object
                    srcFile->setSrcFileHref(filePath);
                    srcFile->setSrcFileName(filePath.filename());
                    srcFile->setPathFromCurrentDirToMainDir(getSource()->getPathFromCurrentDirToMainDir());
                    bf::path absPathToMainDir;
                    bf::path relPathFromMainDir;
                    if (filePath.is_absolute())
                    {
                        //absPathToMainDir is path to the directory with the imported catalog file
                        absPathToMainDir = filePath.parent_path();
                        relPathFromMainDir = bf::path(); // relative path is empty
                    }
                    else
                    {
                        //absPathToMainDir is path to directory with the file with OpenSCENARIO root element
                        absPathToMainDir = getSource()->getAbsPathToMainDir();
						// check if this works!!
                        //relative path is path from directory from absPathToMainDir to the directory with the imported file
                        std::string pathFromExeToMainDir = getParentObj()->getSource()->getPathFromCurrentDirToMainDir().generic_string();
                        std::string tmpRelPathFromMainDir = filePath.parent_path().generic_string();
                        if (pathFromExeToMainDir.empty())
                        {
                            relPathFromMainDir = tmpRelPathFromMainDir;
                        }
                        else
                        {
                            relPathFromMainDir = tmpRelPathFromMainDir.substr(pathFromExeToMainDir.length() + 1);
                        }
                    }
                    srcFile->setAbsPathToMainDir(absPathToMainDir);
                    srcFile->setRelPathFromMainDir(relPathFromMainDir);
                    srcFile->setRootElementName(rootElemName);

                    //object for objectName
                    std::string catalogTypeName = found->second;
                    oscObjectBase *obj = oscFactories::instance()->objectFactory->create(catalogTypeName);
                    if(obj)
                    {
                        obj->initialize(getBase(), this, NULL, srcFile);
                        obj->parseFromXML(rootElem, srcFile);
                        //add objectName and object to oscCatalog map

						ObjectParams *param = &(objectFound)->second;
						param->object = obj;

                        //add sourcFile to vector
                        getBase()->addToSrcFileVec(srcFile);

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
                    std::cerr << "Warning! Can't determine an typeName for catalog " << m_catalogName << std::endl;
                }
 //           }
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

bool oscCatalog::fullReadCatalogObjectFromFile(const bf::path &fileNamePath)
{
    bool success = false;

    std::string name = getObjectNameFromFile(fileNamePath);
    if (name != "")
    {
  //      int objectRefId = successIntVar.second;
        if (addObjToObjectsMap(name, fileNamePath, NULL))
        {
            if (fullReadCatalogObjectWithName(name))
            {
                success = true;
            }
        }
    }
    else
    {
        std::cerr << "Warning! Object for catalog " << m_catalogName << " in " << fileNamePath << " has an empty name and can't be used." << std::endl;
    }

    return success;
}


bool oscCatalog::addCatalogObject(oscObjectBase *objectBase)
{
    if (objectBase)
    {
        //get objectRefId
        //every object in a catalog (e.g. driver, vehicle ...) should have a member 'refId' of type int (oscInt)
        std::string name;

        oscMember *objectNameMember = objectBase->getMembers()["name"];
        if (objectNameMember)
        {
            oscString *objNameVal = dynamic_cast<oscString *>(objectNameMember->getValue());
            name = objNameVal->getValue();
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
            std::cerr << "Error! Can't determine filename and path to write the object." << std::endl;
            return false;
        }

        return addCatalogObject(name, objectBase, fileNamePath);
    }
    else
    {
        std::cerr << "Error! Given pointer to object isn't accessible." << std::endl;
        return false;
    }
}

bool oscCatalog::addCatalogObject(const std::string &name, oscObjectBase *objectBase, const bf::path &fileNamePath)
{

	if (objectBase != NULL && !fileNamePath.empty())
	{
		ObjectsMap::const_iterator foundObjects = m_Objects.find(name);

		if (foundObjects == m_Objects.end())
		{
			//add objectRefId and fileName to m_Objects
			if (addObjToObjectsMap(name, fileNamePath, objectBase))
			{
				return true;
			}
		}
		else
		{
			std::cerr << "Warning: Can't add catalog object name "  << name << ". An object with this name is already registered." << std::endl;
		}
	}
	else
	{
		std::cerr << "Warning! Can't add catalog object with name " << name << ". Empty filename or no pointer to object." << std::endl;
	}

	return false;
}

bool oscCatalog::removeCatalogObject(const std::string &name)
{
    ObjectsMap::const_iterator found = m_Objects.find(name);
    if (found != m_Objects.end())
    {
		bf::path objectPath = found->second.fileName;
		bf::remove(objectPath);
        m_Objects.erase(found);
        return true;
    }
    else
    {
        std::cerr << "Error! Can't remove object with name " << name << " from catalog " << m_catalogName << ". Object not found." << std::endl;
        return false;
    }
}

oscObjectBase *oscCatalog::getCatalogObject(const std::string &name)
{
    ObjectsMap::const_iterator found = m_Objects.find(name);
    if (found != m_Objects.end())
    {
        return found->second.object;
    }
    else
    {
        return NULL;
    }
}

std::string oscCatalog::getPath(const std::string &name)
{
	ObjectsMap::const_iterator found = m_Objects.find(name);
	if (found != m_Objects.end())
	{
		return found->second.fileName.string();
	}
	else
	{
		return NULL;
	}
}

void oscCatalog::writeCatalogToDOM()
{
	for (unordered_map<std::string, ObjectParams>::const_iterator it = m_Objects.begin(); it != m_Objects.end(); it++)
	{
		oscObjectBase *objFromCatalog = it->second.object;
		if (objFromCatalog)
		{
			xercesc::DOMDocument *objFromCatalogXmlDoc = objFromCatalog->getSource()->getOrCreateXmlDoc();
				
			xercesc::DOMElement *rootElement = objFromCatalogXmlDoc->getDocumentElement();
			objFromCatalog->writeToDOM(rootElement, objFromCatalogXmlDoc);
		}
	}
}

void oscCatalog::clearDOM()
{
	for (unordered_map<std::string, ObjectParams>::const_iterator it = m_Objects.begin(); it != m_Objects.end(); it++)
	{
		oscObjectBase *objFromCatalog = it->second.object;
		if (objFromCatalog)
		{
			objFromCatalog->getSource()->clearXmlDoc();
		}
	}
}

void oscCatalog::writeCatalogToDisk()
{
	for (unordered_map<std::string, ObjectParams>::const_iterator it = m_Objects.begin(); it != m_Objects.end(); it++)
	{
		oscObjectBase *objFromCatalog = it->second.object;
		if (objFromCatalog)
		{
			objFromCatalog->getSource()->writeFileToDisk();
		}
	}
}

/*****
 * private functions
 *****/

std::string oscCatalog::getObjectNameFromFile(const bf::path &fileNamePath)
{
//    SuccessIntVar successIntVar = std::make_pair(false, -1);
	std::string  attributeName;
    OpenScenarioBase *oscBase = new OpenScenarioBase;
    bool validate = getBase()->getValidation();
    xercesc::DOMElement *rootElem = oscBase->getRootElement(fileNamePath.generic_string(), m_catalogName, getType(m_catalogName), validate);

    if (rootElem)
    {
        std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());
/*
        if (rootElemName == m_catalogName)
        { */
            //attribute refId is of type int (oscInt) and store the value of objectRefId
            xercesc::DOMAttr *attribute = rootElem->getAttributeNode(xercesc::XMLString::transcode("name"));

            if (attribute)
            {
				std::string attributeName = xercesc::XMLString::transcode(attribute->getName());
  //              successIntVar = getIntFromIntAttribute(attribute);
            }
//        }
    }

    delete oscBase;

    return attributeName;
}

oscCatalog::SuccessIntVar oscCatalog::getIntFromIntAttribute(xercesc::DOMAttr *attribute)
{
    SuccessIntVar successIntVar = std::make_pair(false, -1);
    oscMemberValue::MemberTypes memberTypeInt = oscMemberValue::INT;
    oscMemberValue *memberValInt = oscFactories::instance()->valueFactory->create(memberTypeInt);
    bool initializeSuccess = memberValInt->initialize(attribute);
    if (initializeSuccess)
    {
        oscIntValue *objIntVal = dynamic_cast<oscIntValue *>(memberValInt);
        int intVar = objIntVal->getValue();

        successIntVar = std::make_pair(true, intVar);
    }

    return successIntVar;
}

std::string oscCatalog::generateRefId(int startId)
{
	int refId = startId - 1;
	std::string s;
	ObjectsMap::const_iterator found;
	do
	{
		s = std::to_string(++refId);
		found = m_Objects.find(s);
	}while(found != m_Objects.end());

	return s;
}

bool oscCatalog::parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src, bool saveInclude)
{
	bool result = oscObjectBase::parseFromXML(currentElement,src);
	if(result)
	{
		//catalog type
		std::string catalogType = getOwnMember()->getName();
		catalogType.erase(catalogType.length() - std::string("Catalog").length());
		setCatalogName(catalogType);

		if (Directory.exists())
		{
			//path to Directory
			//object/member is of type oscCatalog and has a member Directory,
			//set variable pathToCatalogDir with type bf::path from std::string
			bf::path pathToCatalogDir(Directory->path.getValue());

			bf::path pathToCatalogDirToUse;
			//check if path is absolute or relative
			if (pathToCatalogDir.is_absolute())
			{
				pathToCatalogDirToUse = pathToCatalogDir;
			}
			else
			{
				pathToCatalogDirToUse = source->getPathFromCurrentDirToMainDir();
				pathToCatalogDirToUse /= pathToCatalogDir;
			}

			//get all catalog object filenames
			std::vector<bf::path> filenames = getXoscFilesFromDirectory(pathToCatalogDirToUse);

			//parse all files
			//store object name and filename in map
			fastReadCatalogObjects(filenames);

			//////
			//enable full read of catalogs in oscTest with argument '-frc'
			//
			if (base->getFullReadCatalogs())
			{
				//generate the objects for this catalog and store them
				for (auto &it : getObjectsMap())
				{
					fullReadCatalogObjectWithName(it.first);
				}
			}
		}
	}
	return result;
}
//
bool oscCatalog::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude)
{
	writeCatalogToDOM();
	return oscObjectBase::writeToDOM(currentElement,document,writeInclude);
}
