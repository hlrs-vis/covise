/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscObjectBase.h"
#include "OpenScenarioBase.h"
#include "oscSourceFile.h"
#include "utilities.h"

#include <iostream>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMImplementation.hpp>


using namespace OpenScenario;


oscObjectBase::oscObjectBase()
{
    base = NULL;
    source = NULL;
    parentObj = NULL;
    ownMem = NULL;
}

oscObjectBase::~oscObjectBase()
{

}


//
void oscObjectBase::initialize(OpenScenarioBase *b, oscObjectBase *parentObject, oscMember *ownMember, oscSourceFile *s)
{
    base = b;
    parentObj = parentObject;
    ownMem = ownMember;
    source = s;
}

void oscObjectBase::addMember(oscMember *m)
{
    members[m->getName()]=m;
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
	if (members.count(s) == 0)
	{
		return NULL;
	}

	return members.at(s);
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
    parentObj = pObj;
}

void oscObjectBase::setOwnMember(oscMember *om)
{
    ownMem = om;
}

oscObjectBase *oscObjectBase::getParentObj() const
{
    return parentObj;
}

oscMember *oscObjectBase::getOwnMember() const
{
    return ownMem;
}


//
bool oscObjectBase::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
{
    for(MemberMap::iterator it = members.begin();it != members.end(); it++)
    {
        oscMember *member = it->second;
        if(member)
        {
            if(member->getType() == oscMemberValue::OBJECT)
            {
                oscObjectBase *obj = member->getObject();

                if (obj)
                {
                    //xml document and element used for writing
                    xercesc::DOMDocument *docToUse;
                    xercesc::DOMElement *elementToUse;

                    //arrayMember
                    oscArrayMember *aMember = dynamic_cast<oscArrayMember *>(member);

                    //xml document for member
                    xercesc::DOMDocument *srcXmlDoc = obj->getSource()->getXmlDoc();

                    //determine document and element for writing
                    //
                    if (document != srcXmlDoc)
                    {
                        //add include element to currentElement and add XInclude namespace to root element of new xml document
                        const XMLCh *fileHref = obj->getSource()->getSrcFileHrefAsXmlCh();
                        addXInclude(currentElement, document, fileHref);

                        //member and arrayMember use a new document and the root element of this document
                        docToUse = srcXmlDoc;
                        elementToUse = docToUse->getDocumentElement();
                    }
                    else
                    {
                        //member and arrayMember use the same document
                        docToUse = document;

                        if (aMember)
                        {
                            //write arrayMember (the container)
                            //(and the arrayMembers are written under this element in the write function)
                            elementToUse = aMember->writeArrayMemberToDOM(currentElement, docToUse);
                        }
                        else
                        {
                            elementToUse = currentElement;
                        }
                    }

                    //write elements into xml documents
                    //
                    if (aMember)
                    {
                        //for arrayMember there is no differentiation if document != srcXmlDoc is true or false
                        for (int i = 0; i < aMember->size(); i++)
                        {
                            std::string aMemChildElemName = aMember->at(i)->getOwnMember()->getName();
                            //obj == object of array member
                            //find the member of this obj and set the value with element of the array vector
                            oscMember *aMemberMember = obj->getMembers().at(aMemChildElemName);
                            aMemberMember->setValue(aMember->at(i));

                            //write array member into root element of new xml document
                            obj->writeToDOM(elementToUse, docToUse);
                        }
                    }
                    else
                    {
                        if (document != srcXmlDoc)
                        {
                            //write members of member into root element of new xml document
                            obj->writeToDOM(elementToUse, docToUse);
                        }
                        else
                        {
                            //oscMember
                            member->writeToDOM(elementToUse, docToUse);
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
                    member->writeToDOM(currentElement,document);
                }
            }
        }
    }

    return true;
}

bool oscObjectBase::parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src)
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
            //only "xml:base" is evaluated during the determination of the source file
            if (attributeName != "xmlns" && attributeName != "xmlns:osc" && attributeName != "xml:base")
            {
                oscMember *m = members[attributeName];
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
                            v->initialize(attribute);
                            m->setValue(v);
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

            oscMember *m = members[memberName];
            if(m)
            {
                std::string memTypeName = m->getTypeName();
                oscSourceFile *srcToUse = determineSrcFile(memberElem, src);

                //oscArrayMember
                //
                oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);
                if(am)
                {
                    //check if array member has attributes
                    //attributes of an array member are not supported
                    xercesc::DOMNamedNodeMap *arrayMemAttributes = memberElem->getAttributes();
                    for (int i = 0; i < arrayMemAttributes->getLength(); i++)
                    {
                        xercesc::DOMAttr *attrib = dynamic_cast<xercesc::DOMAttr *>(arrayMemAttributes->item(i));
                        if(attrib != NULL)
                        {
                            std::string attribName = xercesc::XMLString::transcode(attrib->getName());
                            if (attribName != "xmlns" && attribName != "xmlns:osc" && attribName != "xml:base")
                            {
                                std::cerr << "\n Array member " << memberName << " has attributes." << std::endl;
                                std::cerr << " Attributes of an array member are not supported.\n" << std::endl;
                            }
                        }
                    }

                    //member has no value (value doesn't exist)
                    if ( !m->exists() )
                    {
                        //generate the object for oscArrayMember
                        //(it's the container for the array members)
                        oscObjectBase *objAMCreated = oscFactories::instance()->objectFactory->create(memTypeName);

                        if(objAMCreated)
                        {
                            objAMCreated->initialize(base, this, m, srcToUse);
                            m->setValue(objAMCreated);
                            m->setParentMember(ownMem);
                        }
                        else
                        {
                            std::cerr << "could not create an object arrayMember of type " << memTypeName << std::endl;
                        }
                    }

                    //object for oscArrayMember
                    oscObjectBase *objAM = m->getObject();
                    if(objAM)
                    {
                        xercesc::DOMNodeList *arrayMembersList = memberElem->getChildNodes();

                        //generate the children members and store them in the array
                        for (unsigned int arrayIndex = 0; arrayIndex < arrayMembersList->getLength(); ++arrayIndex)
                        {
                            xercesc::DOMElement *arrayMemElem = dynamic_cast<xercesc::DOMElement *>(arrayMembersList->item(arrayIndex));
                            if (arrayMemElem != NULL)
                            {
                                std::string arrayMemElemName = xercesc::XMLString::transcode(arrayMemElem->getNodeName());

                                oscMember *ame = objAM->getMembers()[arrayMemElemName];
                                std::string arrayMemTypeName = ame->getTypeName();

                                oscObjectBase *obj = oscFactories::instance()->objectFactory->create(arrayMemTypeName);
                                if(obj)
                                {
                                    oscSourceFile *arrayElemSrcToUse = determineSrcFile(arrayMemElem, srcToUse);
                                    obj->initialize(base, objAM, ame, arrayElemSrcToUse);
                                    am->push_back(obj);
                                    ame->setParentMember(objAM->getOwnMember());
                                    obj->parseFromXML(arrayMemElem, arrayElemSrcToUse);
                                }
                                else
                                {
                                    std::cerr << "could not create an object array of type " << arrayMemTypeName << std::endl;
                                }
                            }
                        }
                    }
                }
                //oscMember
                //
                else
                {
                    //member has a value (value exists)
                    if ( m->exists() )
                    {
                        std::cerr << "\n Warning!" << std::endl;
                        std::cerr << "  Member \"" << m->getName() << "\" exists more than once as child of element \"" << xercesc::XMLString::transcode(currentElement->getNodeName()) << "\"" << std::endl;
                        std::cerr << "  Only first entry is used." << std::endl;
                        std::cerr << "  \"" << m->getName() << "\" from file: " << m->getOwner()->getSource()->getSrcFileHrefAsStr() << " is used.\n" << std::endl;
                    }
                    //member has no value (value doesn't exist)
                    else
                    {
                        oscObjectBase *obj = oscFactories::instance()->objectFactory->create(memTypeName);
                        if(obj)
                        {
                            obj->initialize(base, this, m, srcToUse);
                            m->setValue(obj);
                            m->setParentMember(ownMem);
                            obj->parseFromXML(memberElem, srcToUse);
                        }
                        else
                        {
                            std::cerr << "could not create an object member of type " << memTypeName << std::endl;
                        }
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

    return true;
}



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
        fileNamePath *fnPath = newSrc->getFileNamePath(newSrc->getSrcFileHrefAsStr());

        //new srcFileName
        newSrc->setSrcFileName(fnPath->fileName);

        //new mainDocPath and relPathFromMainDoc
        if (base->getSrcFileVec().size() == 1) //only sourceFile of OpenScenario is present
        {
            newSrc->setMainDocPath(base->source->getMainDocPath());
            newSrc->setRelPathFromMainDoc(fnPath->path);
        }
        else
        {
            newSrc->setMainDocPath(srcF->getMainDocPath());

            std::string srcRelPathFromMainDoc = srcF->getRelPathFromMainDoc();
            std::string newSrcRelPathFromMainDoc = fnPath->path;

            std::string relPathFromMainDocToUse;
            if (srcRelPathFromMainDoc == "")
            {
                if (newSrcRelPathFromMainDoc == "")
                {
                    relPathFromMainDocToUse = "";
                }
                else
                {
                    relPathFromMainDocToUse = newSrcRelPathFromMainDoc;
                }
            }
            else
            {
                if (newSrcRelPathFromMainDoc == "")
                {
                    relPathFromMainDocToUse = srcRelPathFromMainDoc;
                }
                else
                {
                    relPathFromMainDocToUse = srcRelPathFromMainDoc + newSrcRelPathFromMainDoc;
                }
            }

            newSrc->setRelPathFromMainDoc(relPathFromMainDocToUse);
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
