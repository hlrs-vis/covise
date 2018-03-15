/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscObjectBase.h"
#include "OpenScenarioBase.h"
#include "oscSourceFile.h"
#include "oscUtilities.h"
#include "oscArrayMember.h"

#include <iostream>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


using namespace OpenScenario;


oscObjectBase::oscObjectBase() :
        base(NULL),
        source(NULL),
        parentObject(NULL),
        ownMember(NULL)
{

}

oscObjectBase::~oscObjectBase()
{

}

/************
* clone members
************/
void oscObjectBase::cloneMembers(oscObjectBase *objectBase, oscObjectBase *parentObj, oscMember *ownMember)
{
	initialize(base, parentObj, ownMember, objectBase->getSource());

	std::list<oscObjectBase::MemberElement> cloneMembers = objectBase->getMembers();
	for (auto it = cloneMembers.cbegin(); it != cloneMembers.cend(); it++ )
	{
		oscMember *cloneMember = (*it).member;
		oscObjectBase *memberObject = cloneMember->getObjectBase();
		oscMember *member = getMember((*it).name);
		if (cloneMember->isSelected())
		{
			member->setSelected(true);
		}

		if (memberObject)
		{
			oscObjectBase *obj = member->createObjectBase();

			obj->cloneMembers(memberObject, this, member);
		}
		else
		{
			oscMemberValue *value = cloneMember->getValue();
			if (value)
			{
				oscMemberValue *val = getMember((*it).name)->createValue();
				if (oscIntValue *iv = dynamic_cast<oscIntValue *>(value))
				{
					val->setValue(iv->getValue());
				}
				else if (oscUIntValue *uv = dynamic_cast<oscUIntValue *>(value))
				{
					val->setValue(uv->getValue());
				}
				else if (oscShortValue *sv = dynamic_cast<oscShortValue *>(value))
				{
					val->setValue(sv->getValue());
				}
				else if (oscUShortValue *usv = dynamic_cast<oscUShortValue *>(value))
				{
					val->setValue(usv->getValue());
				}
				else if (oscFloatValue *fv = dynamic_cast<oscFloatValue *>(value))
				{
					val->setValue(fv->getValue());
				}
				else if (oscDoubleValue *dv = dynamic_cast<oscDoubleValue *>(value))
				{
					val->setValue(dv->getValue());
				}
				else if (oscStringValue *sv = dynamic_cast<oscStringValue *>(value))
				{
					val->setValue(sv->getValue());
				}

				else if (oscBoolValue *iv = dynamic_cast<oscBoolValue *>(value))
				{
					val->setValue(iv->getValue());
				}

				else if (oscDateTimeValue *iv = dynamic_cast<oscDateTimeValue *>(value))
				{
					val->setValue(iv->getValue());

				}
			}
		}

	}

}

//
void oscObjectBase::initialize(OpenScenarioBase *b, oscObjectBase *parentObject, oscMember *ownMember, oscSourceFile *s)
{
    base = b;
    this->parentObject = parentObject;
    this->ownMember = ownMember;
    source = s;

	if (parentObject && parentObject->hasChoice())
	{
		parentObject->setChosenMember(ownMember);
	} 
}


void oscObjectBase::addMember(oscMember *m)
{
	MemberElement element = {m->getName(), m};
	members.push_back(element);
}

void oscObjectBase::setBase(OpenScenarioBase *b)
{
    base = b;
}

void oscObjectBase::setSource(oscSourceFile *s)
{
    source = s;
}

oscObjectBase::MemberMap oscObjectBase::getMembers() const
{
    return members;
}

oscMember *oscObjectBase::getMember(const std::string &s) const
{
	for (auto it = members.cbegin(); it != members.cend(); ++it)
	{
		if ((*it).name == s)
		{
			return (*it).member;
		}
	}

	return NULL;
}

OpenScenarioBase *oscObjectBase::getBase() const
{
    return base;
}

oscSourceFile *oscObjectBase::getSource() const
{
    return source;
}


//
void oscObjectBase::setParentObj(OpenScenarioBase *pObj)
{
    parentObject = pObj;
}

void oscObjectBase::setOwnMember(oscMember *om)
{
    ownMember = om;
}

oscObjectBase *oscObjectBase::getParentObj() const
{
    return parentObject;
}

oscMember *oscObjectBase::getOwnMember() const
{
    return ownMember;
}

//
bool oscObjectBase::hasChoice() const
{
	for (auto it = members.cbegin(); it != members.cend(); ++it)
	{
		oscMember *member = (*it).member;
		if (member->isChoice())
		{
			return true;
		}
	}

	return false;
}

std::list<oscMember *> oscObjectBase::getChoice() const
{
	std::list<oscMember *> mc;

	for (auto it = members.cbegin(); it != members.cend(); ++it)
	{
		oscMember *m = (*it).member;
		if (m->isChoice())
		{
			mc.push_back(m);
		}
	}

	return mc;
}

void oscObjectBase::setChosenMember(oscMember *chosenMember)
{
	unsigned short choiceNumber = chosenMember->choiceNumber();
	for (auto it = members.cbegin(); it != members.cend(); ++it)
	{
		oscMember *member = (*it).member;

		if (member->isSelected() && (member->choiceNumber() == choiceNumber))
		{
				member->setSelected(false);
		}
	}

	chosenMember->setSelected(true);
}

oscMember *oscObjectBase::getChosenMember(unsigned short choiceNumber)
{
	for (auto it = members.cbegin(); it != members.cend(); ++it)
	{
		oscMember *member = (*it).member;

		if (member->isSelected() && (member->choiceNumber() == choiceNumber))
		{
				return member;
		}
	}

	return NULL;
}

//
void oscObjectBase::addMemberToOptional(oscMember *m)
{
    optional.push_back(m);
}

bool oscObjectBase::hasOptional() const
{
    return (optional.size() > 0) ? true : false;
}

oscObjectBase::MemberOptional oscObjectBase::getOptional() const
{
    return optional;
}

bool oscObjectBase::isMemberOptional(oscMember *m)
{
	for (auto it = optional.cbegin(); it != optional.cend(); it++)
	{
		if (*it == m)
		{
			return true;
		}
	}

	return false;
}

//
oscObjectBase *oscObjectBase::getObjectByName(const std::string &name)
{
	oscMember *member = getMember(name);
	if (member)
	{
		return member->getOrCreateObjectBase();
	}

	return NULL;
}


//
bool oscObjectBase::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude)
{
	for(MemberMap::iterator it = members.begin();it != members.end(); it++)
	{
		oscMember *member = it->member;
		if (member)
		{
			bool isChoice = member->isChoice();
			if(!isChoice || (isChoice && member->isSelected()))
			{
				if(member->getType() == oscMemberValue::OBJECT)
				{
					//ArrayMember
					oscArrayMember *aMember = dynamic_cast<oscArrayMember *>(member);
					if (aMember != NULL)
					{
						for(oscArrayMember::iterator it =aMember->begin();it != aMember->end();it++)
						{
							oscObjectBase *obj = *it;

							//xml document for member
							xercesc::DOMDocument *srcXmlDoc = obj->getSource()->getXmlDoc();

							//determine document and element for writing
							//
							if (srcXmlDoc && (document != srcXmlDoc) && writeInclude)
							{
								//add include element to currentElement and add XInclude namespace to root element of new xml document
								const XMLCh *fileHref = obj->getSource()->getSrcFileHrefAsXmlCh();
								addXInclude(currentElement, document, fileHref);

								//member and ArrayMember use a new document and the root element of this document
								//write members of member into root element of new xml document
								obj->writeToDOM(srcXmlDoc->getDocumentElement(), srcXmlDoc, writeInclude);
							}
							else
							{
								//oscMember

								xercesc::DOMElement *memberElement = document->createElement(xercesc::XMLString::transcode(member->getName().c_str()));
								currentElement->appendChild(memberElement);
								obj->writeToDOM(memberElement, document, writeInclude);
							}
						}
					}
					else
					{
						oscObjectBase *obj = member->getObjectBase();
						if (obj)
						{
							//xml document for member
							xercesc::DOMDocument *srcXmlDoc = obj->getSource()->getXmlDoc();

							//determine document and element for writing
							//
							if (srcXmlDoc && (document != srcXmlDoc) && writeInclude)
							{
								//add include element to currentElement and add XInclude namespace to root element of new xml document
								const XMLCh *fileHref = obj->getSource()->getSrcFileHrefAsXmlCh();
								addXInclude(currentElement, document, fileHref);

								//member and ArrayMember use a new document and the root element of this document
								//write members of member into root element of new xml document
								obj->writeToDOM(srcXmlDoc->getDocumentElement(), srcXmlDoc, writeInclude);
							}
							else
							{
								//oscMember
								member->writeToDOM(currentElement, document, writeInclude);
							}

						}
					}
				}
				else
				{
					oscArrayMember *am = dynamic_cast<oscArrayMember *>(member);
					if(am)
					{
						std::cerr << "Array values not yet implemented" << std::endl;
					}
					else
					{
						member->writeToDOM(currentElement,document,writeInclude);
					}
				}
			}
		}
    }

    return true;
}

bool oscObjectBase::parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src, bool saveInclude)
{
    xercesc::DOMNodeList *membersList = currentElement->getChildNodes();
    xercesc::DOMNamedNodeMap *attributes = currentElement->getAttributes();

    //Attributes of current element
    //
    for (unsigned int attrIndex = 0; attrIndex < attributes->getLength(); ++attrIndex)
    {
        xercesc::DOMAttr *attribute = dynamic_cast<xercesc::DOMAttr *>(attributes->item(attrIndex));
        if(attribute != NULL)
        {
            std::string attributeName = xercesc::XMLString::transcode(attribute->getName());

            //attributes "xmlns", "xmlns:osc" and "xml:base" are generated and/or used with namespaces and XInclude
            //they have no representation in the object structure
            //"xml:base" is only evaluated during the determination of the source file
            if (attributeName != "xmlns" && attributeName != "xmlns:osc" && attributeName != "xml:base")
            {
                oscMember *m = getMember(attributeName);
                if(m)
                {
                    oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);
                    if(am)
                    {
                        std::cerr << "Array values not yet implemented" << std::endl;
                    }
                    else
                    {
                        oscMemberValue::MemberTypes attributeType = m->getType();
                        oscMemberValue *v = oscFactories::instance()->valueFactory->create(attributeType);
                        if(v)
                        {
                            if(attributeType == oscMemberValue::ENUM)
                            {
                                oscEnumValue *ev = dynamic_cast<oscEnumValue *>(v);
                                oscEnum *em = dynamic_cast<oscEnum *>(m);
                                if(ev && em)
                                {
                                    ev->enumType = em->enumType;
                                }
                            }

                            bool initializeSuccess = v->initialize(attribute);
                            if (initializeSuccess)
                            {
                                m->setValue(v);
                            }
                            else
                            {
                                std::cerr << " No value set for attribute '" << attributeName << "' in element '" << this->getOwnMember()->getName() << "'." << std::endl;
                            }
                        }
                        else
                        {
                            std::cerr << "could not create a value of type " << attributeType << std::endl;
                        }
                    }
                }
                else
                {
                    std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member value called " << attributeName << std::endl;
                }
            }
        }
    }

    //children of current element
    //
    for (unsigned int memberIndex = 0; memberIndex < membersList->getLength(); ++memberIndex)
    {
        xercesc::DOMElement *memberElem = dynamic_cast<xercesc::DOMElement *>(membersList->item(memberIndex));
        if(memberElem != NULL)
        {
            std::string memberName = xercesc::XMLString::transcode(memberElem->getNodeName());

            oscMember *m = getMember(memberName);
            if(m)
            {
                std::string memTypeName = m->getTypeName();

				oscSourceFile *srcToUse = src;
				if (saveInclude)
				{
					srcToUse = determineSrcFile(memberElem, src);
				}

				oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);

				//member has a value (value exists)
				if (am == NULL &&  m->exists())
				{
					std::cerr << "\n Warning!" << std::endl;
					std::cerr << "  Member \"" << m->getName() << "\" exists more than once as child of element \"" << xercesc::XMLString::transcode(currentElement->getNodeName()) << "\"" << std::endl;
					std::cerr << "  Only first entry is used." << std::endl;
					std::cerr << "  \"" << m->getName() << "\" from file: " << m->getOwner()->getSource()->getSrcFileHrefAsStr() << " is used.\n" << std::endl;
				}
				//member has no value (value doesn't exist)
				else
				{
					std::string className = nameMapping::instance()->getClassName(memTypeName, std::string(getScope()));
					oscObjectBase *obj = oscFactories::instance()->objectFactory->create(className);
					if (obj)
					{
						obj->initialize(base, this, m, srcToUse);
						if (am != NULL)
						{
							am->push_back(obj);
						}
						else
						{
							m->setValue(obj);
						}
						m->setParentMember(ownMember);
						obj->parseFromXML(memberElem, srcToUse, saveInclude);
					}
					else
					{
						std::cerr << "could not create an object member of type " << memTypeName << std::endl;
					}
                }
			}
            //no member
            //
            else
            {
                std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member called " << memberName << std::endl;
            }
        }
    }
	finishedParsing();
    return true;
}

bool oscObjectBase::writeToDisk()
{

	xercesc::DOMDocument *objFromCatalogXmlDoc = source->getOrCreateXmlDoc();
	xercesc::DOMElement *rootElement = objFromCatalogXmlDoc->getDocumentElement();
	if (!writeToDOM(rootElement, objFromCatalogXmlDoc))
	{
		return false;
	}
	
	if (!source->writeFileToDisk())
	{
		return false;
	}

	source->clearXmlDoc();

	return true;
}

oscObjectBase *oscObjectBase::readDefaultXMLObject(bf::path destFilePath, const std::string &memberName, const std::string &typeName, oscSourceFile *srcFile)
{
	oscObjectBase *obj = NULL;
	OpenScenarioBase *oscBase = new OpenScenarioBase;

	//in readDefaultXMLObject no validation should be done,
	//because default objects are valid
	xercesc::DOMElement *rootElem = oscBase->getDefaultXML(typeName);
	if (rootElem)
	{
		std::string rootElemName = xercesc::XMLString::transcode(rootElem->getNodeName());

		if (rootElemName == memberName)
		{
			bool newSourceFile = false;
			if (!srcFile)
			{
				//sourceFile for objectName
				srcFile = new oscSourceFile();

				//set variables for srcFile, differentiate between absolute and relative path for catalog object
				srcFile->setSrcFileHref(destFilePath);
				srcFile->setSrcFileName(destFilePath.filename());
				srcFile->setPathFromCurrentDirToMainDir(getSource()->getPathFromCurrentDirToMainDir());
				bf::path absPathToMainDir;
				bf::path relPathFromMainDir;
				if (destFilePath.is_absolute())
				{
					//absPathToMainDir is path to the directory with the imported catalog file
					absPathToMainDir = destFilePath.parent_path();
					relPathFromMainDir = bf::path(); // relative path is empty
				}
				else
				{
					//absPathToMainDir is path to directory with the file with OpenSCENARIO root element
					absPathToMainDir = getSource()->getAbsPathToMainDir();
					// check if this works!!
					//relative path is path from directory from absPathToMainDir to the directory with the imported file
					std::string pathFromExeToMainDir = getParentObj()->getSource()->getPathFromCurrentDirToMainDir().generic_string();
					std::string tmpRelPathFromMainDir = destFilePath.parent_path().generic_string();
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

				newSourceFile = true;
			}

			//object for objectName

			std::string className = nameMapping::instance()->getClassName(typeName, std::string(getScope()));
			obj = oscFactories::instance()->objectFactory->create(className);
			if(obj)
			{
				obj->initialize(getBase(), NULL, NULL, srcFile);
				obj->parseFromXML(rootElem, srcFile, false);

				//add sourcFile to vector
				getBase()->addToSrcFileVec(srcFile);

			}
			else if (newSourceFile)
			{
				std::cerr << "Error! Could not create an object member of type " << typeName << std::endl;
				delete srcFile;
			}
		}
	}

	delete oscBase;

	return obj;

}

unsigned char oscObjectBase::validate()
{
	for(MemberMap::iterator it = members.begin();it != members.end(); it++)
	{
		oscMember *member = it->member;

		if (!member->validate())
		{
			return VAL_invalid;
		}	
	}

	return VAL_valid;
}

/* void oscObjectBase::validate(std::string *errorMessage)
{
	// write temporary file
	//
	bf::path tmpFilename = bf::temp_directory_path() / bf::path("tmpValidate.xosc");
	std::cerr << tmpFilename << std::endl;

	xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();

	std::string name = ownMember->getName();

	const XMLCh *source = xercesc::XMLString::transcode(name.c_str());
	xercesc::DOMDocument *xmlSrcDoc = impl->createDocument(0, source, 0);
	if (xmlSrcDoc)
	{
		writeToDOM(xmlSrcDoc->getDocumentElement(), xmlSrcDoc, false);


#if (XERCES_VERSION_MAJOR < 3)
		xercesc::DOMWriter *writer = impl->createDOMWriter();
		xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(tmpFilename.generic_string().c_str());
		if (!writer->writeNode(xmlTarget, xmlSrcDoc->getDocumentElement()))

		{
			std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
			delete xmlTarget;
			delete writer;
			return false;
		}

#else
		xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
		// set the format-pretty-print feature
		if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
		{
			writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
		}

		xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(tmpFilename.generic_string().c_str());

		xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
		output->setByteStream(xmlTarget);

		if (!writer->write(xmlSrcDoc, output))
		{
			std::string msg = "OpenScenarioBase::writeXosc: Could not open temporary file for writing!";
			errorMessage = &msg;
			std::cerr << errorMessage << std::endl;
			delete output;
			delete xmlTarget;
			delete writer;
			delete xmlSrcDoc;

			return;
		}

		delete output;
#endif

		delete xmlTarget;
		delete writer;
		delete xmlSrcDoc;

		// validate temporaryFile
		//
		OpenScenarioBase *oscBase = new OpenScenarioBase;
		oscBase->getRootElement(tmpFilename.string(), name,ownMember->getTypeName(), true, errorMessage); 
		delete oscBase;

		try
		{
			bf::remove(tmpFilename);
		}
		catch(...)
		{
			std::cout << tmpFilename << std::endl;
		}

	}
} */


/*****
 * private functions
 *****/

void oscObjectBase::addXInclude(xercesc::DOMElement *currElem, xercesc::DOMDocument *doc, const XMLCh *fileHref)
{
    //add include element 
    const XMLCh *xInclude = xercesc::XMLString::transcode("osc:include");
    xercesc::DOMElement *xIncludeElem = doc->createElement(xInclude);
    const XMLCh *attrHrefName = xercesc::XMLString::transcode("href");
    xIncludeElem->setAttribute(attrHrefName, fileHref);
    currElem->appendChild(xIncludeElem);

    //write namespace for XInclude as attribute to doc root element
    const XMLCh *attrXIncludeNsName = xercesc::XMLString::transcode("xmlns:osc");
    xercesc::DOMElement *docRootElem = doc->getDocumentElement();
    xercesc::DOMAttr *attrNodeXIncludeNs = docRootElem->getAttributeNode(attrXIncludeNsName);
    if (!attrNodeXIncludeNs)
    {
        //XInclude defines a namespace associated with the URI http://www.w3.org/2001/XInclude
        //it is no link, it is treated as a normal string (as a formal identifier)
        const XMLCh *attrXIncludeNsValue = xercesc::XMLString::transcode("http://www.w3.org/2001/XInclude");
        docRootElem->setAttribute(attrXIncludeNsName, attrXIncludeNsValue);
    }
}

oscSourceFile *oscObjectBase::determineSrcFile(xercesc::DOMElement *memElem, oscSourceFile *srcF)
{
    oscSourceFile *srcToUse;

    //if attribute with attrXmlBase is present, the element and all its children were read from a different file
    //therefore we generate a new oscSourceFile
    const XMLCh *attrXmlBase = xercesc::XMLString::transcode("xml:base");
    xercesc::DOMAttr *memElemAttrXmlBase = memElem->getAttributeNode(attrXmlBase);
    if (memElemAttrXmlBase)
    {
        oscSourceFile *newSrc = new oscSourceFile();

        //new srcFileHref
        newSrc->setSrcFileHref(memElemAttrXmlBase->getValue());

        //filename and path
        bf::path fnPath = newSrc->getFileNamePath(newSrc->getSrcFileHrefAsStr());

        //new srcFileName
        newSrc->setSrcFileName(fnPath.filename());

        //new pathFromExeToMainDir, absPathToMainDir and relPathFromMainDir
        if (base->getSrcFileVec().size() == 1) //only sourceFile of OpenScenario is present
        {
            newSrc->setPathFromCurrentDirToMainDir(base->source->getPathFromCurrentDirToMainDir());
            newSrc->setAbsPathToMainDir(base->source->getAbsPathToMainDir());
            newSrc->setRelPathFromMainDir(fnPath.parent_path());
        }
        else
        {
            newSrc->setPathFromCurrentDirToMainDir(srcF->getPathFromCurrentDirToMainDir());
            newSrc->setAbsPathToMainDir(srcF->getAbsPathToMainDir());

            bf::path srcRelPathFromMainDir = srcF->getRelPathFromMainDir();
            bf::path newSrcRelPathFromMainDir = fnPath.parent_path();

            bf::path relPathFromMainDirToUse;
            if (srcRelPathFromMainDir.empty())
            {
                if (newSrcRelPathFromMainDir.empty())
                {
                    relPathFromMainDirToUse = bf::path();
                }
                else
                {
                    relPathFromMainDirToUse = newSrcRelPathFromMainDir;
                }
            }
            else
            {
                if (newSrcRelPathFromMainDir.empty())
                {
                    relPathFromMainDirToUse = srcRelPathFromMainDir;
                }
                else
                {
                    relPathFromMainDirToUse = srcRelPathFromMainDir;
                    relPathFromMainDirToUse /= newSrcRelPathFromMainDir;
                }
            }

            newSrc->setRelPathFromMainDir(relPathFromMainDirToUse);
        }

        //new rootElementName
        newSrc->setRootElementName(memElem->getNodeName());

        base->addToSrcFileVec(newSrc);
        srcToUse = newSrc;
    }
    else
    {
        srcToUse = srcF;
    }

    return srcToUse;
}
