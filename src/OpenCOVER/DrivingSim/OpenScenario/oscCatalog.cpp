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


oscCatalogFile *oscCatalog::getCatalogFile(int index)
{
	if (xoscFiles.size() <= index)
	{
		return NULL;
	}
	return xoscFiles[index];
}
oscCatalogFile *oscCatalog::getCatalogFile(std::string &catalogName, std::string &path)
{
	std::string filename = catalogName + ".xosc";
	for (size_t i = 0; i < xoscFiles.size(); i++)
	{
		if (xoscFiles[i]->getPath() == filename)
		{
			return xoscFiles[i];
		}
	}

	oscCatalogFile *cat = new oscCatalogFile(catalogName,filename,path);
	xoscFiles.push_back(cat);
	return cat;
}

/*****
 * public functions
 *****/

void oscCatalog::getXoscFilesFromDirectory()
{

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
		try
		{
			if (bf::exists(pathToCatalogDirToUse))
			{
				if (bf::is_directory(pathToCatalogDirToUse))
				{
					//bf::recursive_directory_iterator() constructs the end iterator
					for (bf::recursive_directory_iterator it(pathToCatalogDirToUse); it != bf::recursive_directory_iterator(); it++)
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
								std::string catName= contentName.stem().string();
								std::string fileName= contentName.filename().string();
								std::string path = pathToCatalogDirToUse.string();
								oscCatalogFile *file = new oscCatalogFile(catName, fileName, path);
								file->setPath(contentName);
								xoscFiles.push_back(file);
							}
						}
					}
				}
				else
				{
					std::cerr << "Warning! " << pathToCatalogDirToUse << " is not a path to a directory." << std::endl;
				}
			}
			else
			{
				std::cerr << "Warning! File or directory " << pathToCatalogDirToUse << " do not exist." << std::endl;
			}
		}
		catch (const bf::filesystem_error& fse)
		{
			std::cerr << "getXoscFilesFromDirectory(): " << fse.what() << std::endl;
		}
	}
	else
	{
		std::cerr << "Warning! no Directory element available." << std::endl;
	}

}

void oscCatalog::clearAllCatalogs()
{
	xoscFiles.clear();
	m_Objects.clear();
}

void oscCatalog::fastReadCatalogObjects()
{
	clearAllCatalogs();
	getXoscFilesFromDirectory();
    OpenScenarioBase *oscBase = new OpenScenarioBase();
    bool validate = getBase()->getValidation();

    for (size_t i = 0; i < xoscFiles.size(); i++)
    {
        xercesc::DOMElement *rootElem = oscBase->getRootElement(xoscFiles[i]->getPath().generic_string(), m_catalogName, m_catalogType, validate);

        if (rootElem)
        {
			std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

			/*           if (rootElemName == m_catalogName)
			{ */
			xercesc::DOMNodeList *list = rootElem->getElementsByTagName(xercesc::XMLString::transcode(m_catalogName.c_str()));

			for(int it = 0;it<list->getLength();it++)
			{
				xercesc::DOMNode *node = list->item(it);
				if (node)
				{
					xercesc::DOMNamedNodeMap *attributes = node->getAttributes();
					xercesc::DOMNode *attribute = attributes->getNamedItem(xercesc::XMLString::transcode("name"));


					if (attribute)
					{
						
						std::string attributeName = xercesc::XMLString::transcode(attribute->getNodeValue());
						ObjectsMap::const_iterator found = m_Objects.find(attributeName);

						if (found != m_Objects.end())
						{
							std::cerr << "Warning! Object for catalog " << m_catalogName << " with name " << attributeName << " from " << xoscFiles[i]->getPath() << " is ignored." << std::endl;
							std::cerr << "First appearance from file " << found->second.xoscFile->getPath() << " is used." << std::endl;
						}
						else
						{
							ObjectParams param = { xoscFiles[i], NULL};
							m_Objects.emplace(attributeName, param);
						}
					}
					else
					{
						std::cerr << "Warning! Can't find an object for catalog " << m_catalogName << " in file " << xoscFiles[i]->getPath() << " with attribute 'attributeName'." << std::endl;
					}
				}
			}
        }
    }

    delete oscBase;
}


//
void oscCatalog::setCatalogNameAndType(const std::string &catalogName)
{
    m_catalogName = catalogName;
	m_catalogType = "osc" + catalogName;
}

std::string oscCatalog::getCatalogName() const
{
    return m_catalogName;
}

std::string oscCatalog::getCatalogType() const
{
    return m_catalogType;
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
bool oscCatalog::addObjToObjectsMap(const std::string &name, oscCatalogFile *catf, oscObjectBase *object)
{
    ObjectsMap::const_iterator found = m_Objects.find(name);
    if (found != m_Objects.end())
    {
		std::cerr << "Error! Object with name " << name << " exists and is defined in file " << found->second.xoscFile->getPath().string() << std::endl;
        return false;
    }
	catf->addObject(object);
	ObjectParams params = { catf, object};
    std::pair<ObjectsMap::const_iterator, bool> returnVal = m_Objects.emplace(name, params);
    if (returnVal.second == false)
    {
        std::cerr << "Error! Can't insert name " << name << " from file " << catf->getPath().filename().generic_string() << "into map of available objects." << std::endl;
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

oscCatalogFile::oscCatalogFile(std::string &catalogName, std::string &filename, std::string &path)
{		
	srcFile = new oscSourceFile();
	srcFile->setNameAndPath(filename, "OpenSCENARIO", path);
    catalogName = catalogName;
	setPath(filename);
	m_Header = NULL;
}
oscCatalogFile::~oscCatalogFile()
{		
	delete srcFile;
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

	oscCatalogFile *fileObject = objectFound->second.xoscFile;
    if (fileObject != NULL)
    {
        bool success = false;

        OpenScenarioBase *oscBase = new OpenScenarioBase;

		//in fullReadCatalogObjectWithName no validation should be done,
		//because during fastReadCatalogObjects validation is done
		bf::path filePath = fileObject->getPath();
		xercesc::DOMElement *rootElem = oscBase->getRootElement(filePath.generic_string(), m_catalogName, m_catalogType, false);
		if (rootElem)
		{
			std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());


			//set variables for srcFile, differentiate between absolute and relative path for catalog object
			fileObject->srcFile->setSrcFileHref(filePath);
			fileObject->srcFile->setSrcFileName(filePath.filename());
			fileObject->srcFile->setPathFromCurrentDirToMainDir(getSource()->getPathFromCurrentDirToMainDir());
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
			fileObject->srcFile->setAbsPathToMainDir(absPathToMainDir);
			fileObject->srcFile->setRelPathFromMainDir(relPathFromMainDir);
			fileObject->srcFile->setRootElementName(rootElemName);

			xercesc::DOMNodeList *headerList = rootElem->getElementsByTagName(xercesc::XMLString::transcode("FileHeader"));

			//object for header
			fileObject->m_Header = dynamic_cast<oscFileHeader *>(oscFactories::instance()->objectFactory->create("oscFileHeader"));
			if (fileObject->m_Header)
			{
				fileObject->m_Header->initialize(getBase(), this, NULL, fileObject->srcFile);
				fileObject->m_Header->parseFromXML(dynamic_cast<xercesc::DOMElement *>(headerList->item(0)), fileObject->srcFile);
			}

			xercesc::DOMNodeList *catalogList = rootElem->getElementsByTagName(xercesc::XMLString::transcode("Catalog"));
			xercesc::DOMNamedNodeMap *attributes = catalogList->item(0)->getAttributes();
			xercesc::DOMNode *attribute = attributes->getNamedItem(xercesc::XMLString::transcode("name"));


			if (attribute)
			{
				fileObject->catalogName = xercesc::XMLString::transcode(attribute->getNodeValue());
			}

			xercesc::DOMNodeList *list = rootElem->getElementsByTagName(xercesc::XMLString::transcode(m_catalogName.c_str()));

			for(int it = 0;it<list->getLength();it++)
			{
				xercesc::DOMNode *node = list->item(it);
				if (node)
				{
					xercesc::DOMNamedNodeMap *attributes = node->getAttributes();
					xercesc::DOMNode *attribute = attributes->getNamedItem(xercesc::XMLString::transcode("name"));


					if (attribute)
					{

						std::string attributeName = xercesc::XMLString::transcode(attribute->getNodeValue());
						if(attributeName == name)
						{

							//               CatalogTypeTypeNameMap::const_iterator found = s_catalogNameToTypeName.find(m_catalogName);
							//              if (found != s_catalogNameToTypeName.end())
							//             {

							//object for objectName
							oscObjectBase *obj = oscFactories::instance()->objectFactory->create("osc"+m_catalogName);
							if(obj)
							{
								obj->initialize(getBase(), this, NULL, fileObject->srcFile);
								obj->parseFromXML(dynamic_cast<xercesc::DOMElement *>(node), fileObject->srcFile);
								//add objectName and object to oscCatalog map

								ObjectParams *param = &(objectFound)->second;
								param->object = obj;
								fileObject->addObject(obj);

								//add sourcFile to vector
								getBase()->addToSrcFileVec(fileObject->srcFile);

								success = true;
							}
							else
							{
								std::cerr << "Error! Could not create an object member of type " << m_catalogType << std::endl;
							}
						}						
					}
				}
			}
		}
	

        delete oscBase;

        return success;
    }
    else
    {
        std::cerr << "Could not read Catalog " << std::endl;
        return false;
    }
}
/*
bool oscCatalog::addCatalogObject(oscObjectBase *objectBase)
{
    if (objectBase)
    {
        //get objectRefId
        //every object in a catalog (e.g. driver, vehicle ...) should have a member 'refId' of type int (oscInt)
        std::string name;

        oscMember *objectNameMember = objectBase->getMember(name);
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
*/
bool oscCatalog::addCatalogObject(const std::string &name, oscObjectBase *objectBase, oscCatalogFile *catf)
{

	if (objectBase != NULL )
	{
		ObjectsMap::const_iterator foundObjects = m_Objects.find(name);

		if (foundObjects == m_Objects.end())
		{
			//add objectRefId and fileName to m_Objects
			if (addObjToObjectsMap(name, catf, objectBase))
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

void oscCatalogFile::removeObject(oscObjectBase *obj)
{
	 for (auto it = objects.begin(); it != objects.end(); it++)
	 {
		 if(*it == obj)
		 {
			 objects.erase(it);
			 break;
		 }
	 }
}
/*
bool oscCatalog::removeCatalogObject(const std::string &name)
{
    ObjectsMap::const_iterator found = m_Objects.find(name);
    if (found != m_Objects.end())
    {
		bf::path objectPath = found->second.xoscFile->removeObject(found->second.object);
        return true;
    }
    else
    {
        std::cerr << "Error! Can't remove object with name " << name << " from catalog " << m_catalogName << ". Object not found." << std::endl;
        return false;
    }
}*/

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

oscObjectBase *oscCatalog::getCatalogObject(const std::string &catalogName, const std::string &entryName) ///< return pointer to oscObjectBase 
{
	ObjectsMap::const_iterator found = m_Objects.find(entryName);
	if (found != m_Objects.end())
	{
		if (!found->second.object)
		{
			fullReadCatalogObjectWithName(found->first);
		}
		return found->second.object;
	}
	else
	{
		return NULL;
	}
	//TODO search in catalog
	/*
	fullReadCatalogObjectWithName(it->first);
	for (size_t i = 0; i < xoscFiles.size(); i++)
	{
		if (xoscFiles[i]->catalogName == catalogName)
		{
			return xoscFiles[i];
		}
	}
	}*/
}

std::string oscCatalog::getPath(const std::string &name)
{
	ObjectsMap::const_iterator found = m_Objects.find(name);
	if (found != m_Objects.end())
	{
		return found->second.xoscFile->getPath().string();
	}
	else
	{
		return NULL;
	}
}

std::string 
oscCatalog::getObjectPath(OpenScenario::oscObjectBase *object)
{
	 for (auto it = m_Objects.begin(); it != m_Objects.end(); it++)
	 {
		 ObjectParams params = it->second;
		 if (params.object == object)
		 {
			 return params.xoscFile->getPath().string();
		 }
	 }

	 return std::string();
}

OpenScenario::oscObjectBase *
oscCatalog::getObjectfromPath(const std::string &path)
{
	 for (auto it = m_Objects.begin(); it != m_Objects.end(); it++)
	 {
		 ObjectParams params = it->second;
		 if (!params.object)
		 {
			 fullReadCatalogObjectWithName(it->first);
		 }

		 if (params.xoscFile->getPath().string() == path)
		 {
			 return params.object;
		 }
	 }

	 return NULL;
}


void oscCatalog::writeCatalogsToDOM()
{
	for (size_t i = 0; i < xoscFiles.size(); i++)
	{
		xoscFiles[i]->writeCatalogToDOM();
	}

}
void oscCatalogFile::writeCatalogToDOM()
{
	xercesc::DOMElement *catalogElement;
	xercesc::DOMDocument *objFromCatalogXmlDoc = srcFile->getOrCreateXmlDoc();
	if (objects.size() > 0)
	{
		oscObjectBase *obj =objects[0];
		if (obj)
		{
			xercesc::DOMElement *rootElement = objFromCatalogXmlDoc->getDocumentElement();

			xercesc::DOMElement *fhElement = objFromCatalogXmlDoc->createElement(xercesc::XMLString::transcode("FileHeader"));
			rootElement->appendChild(fhElement);
			if (m_Header == NULL)
				m_Header = new oscFileHeader();
			m_Header->writeToDOM(fhElement, objFromCatalogXmlDoc);

			catalogElement = objFromCatalogXmlDoc->createElement(xercesc::XMLString::transcode("Catalog"));
			rootElement->appendChild(catalogElement);

			catalogElement->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode(catalogName.c_str()));
		}
	}

	for (size_t i = 0; i < objects.size(); i++)
	{
		oscObjectBase *objFromCatalog =objects[i];
		if (objFromCatalog)
		{
			std::string catalogTypeName = "";

			if (dynamic_cast<oscVehicle *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Vehicle";
			}
			else if (dynamic_cast<oscDriver *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Driver";
			}
			else if (dynamic_cast<oscPedestrian *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Pedestrian";
			}
			else if (dynamic_cast<oscPedestrianController *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "PedestrianController";
			}
			else if (dynamic_cast<oscMiscObject *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "MiscObject";
			}
			else if (dynamic_cast<oscEnvironment *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Environment";
			}
			else if (dynamic_cast<oscManeuver *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Maneuver";
			}
			else if (dynamic_cast<oscTrajectory *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Trajectory";
			}
			else if (dynamic_cast<oscRoute *>(objFromCatalog) != NULL)
			{
				catalogTypeName = "Route";
			}

			xercesc::DOMElement *catalogItemElement = objFromCatalogXmlDoc->createElement(xercesc::XMLString::transcode(catalogTypeName.c_str()));
			catalogElement->appendChild(catalogItemElement);
			objFromCatalog->writeToDOM(catalogItemElement, objFromCatalogXmlDoc);
		}
	}
}


void oscCatalog::clearDOMs()
{
	for (size_t i = 0; i < xoscFiles.size(); i++)
	{
		xoscFiles[i]->srcFile->clearXmlDoc();
	}
}

void oscCatalog::writeCatalogsToDisk()
{
	writeCatalogsToDOM();
	for (size_t i = 0; i < xoscFiles.size(); i++)
	{
		if(xoscFiles[i]->srcFile->getXmlDoc() != NULL)
		{ 
		    xoscFiles[i]->srcFile->writeFileToDisk();
		}
	}
	clearDOMs();
}

/*****
 * private functions
 *****/

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

const bf::path &oscCatalogFile::getPath()
{
	return fileName;
}

void oscCatalogFile::setPath(const bf::path &fn)
{
	fileName = fn;
}
void oscCatalogFile::addObject(oscObjectBase *obj)
{
	objects.push_back(obj);
}

std::string oscCatalog::generateRefId()
{
	int refId = 0;
	std::string s;
	ObjectsMap::const_iterator found;
	do
	{
		s = m_catalogName + std::to_string(refId++);
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
		setCatalogNameAndType(catalogType);


		//parse all files
		//store object name and filename in map
		fastReadCatalogObjects();

		//////
		//enable full read of catalogs in oscTest with argument '-frc'
		//
		if (base->getFullReadCatalogs())
		{
			//TODO::FIX
			//generate the objects for this catalog and store them
			for (auto &it : getObjectsMap())
			{
				fullReadCatalogObjectWithName(it.first);
			}
		}
	}
	return result;
}
//
bool oscCatalog::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude)
{
	writeCatalogsToDOM();
	return oscObjectBase::writeToDOM(currentElement,document,writeInclude);
}
