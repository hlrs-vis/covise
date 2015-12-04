/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscObjectBase.h>
#include <oscArrayMember.h>
#include <oscObjectVariable.h>
#include <OpenScenarioBase.h>
#include "oscSourceFile.h"

#include <iostream>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/util/XMLString.hpp>


using namespace OpenScenario;


oscObjectBase::oscObjectBase()
{
    base = NULL;
    source = NULL;
}
oscObjectBase::~oscObjectBase()
{
}

void oscObjectBase::initialize(OpenScenarioBase *b, oscSourceFile *s)
{
    base = b;
    source = s;
}

void oscObjectBase::addMember(oscMember *m)
{
    members[m->getName()]=m;
}


bool oscObjectBase::writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document)
{
    for(MemberMap::iterator it = members.begin();it != members.end();it++)
    {
        oscMember *member = it->second;
        if(member)
        {
            if(member->getType() == oscMemberValue::OBJECT)
            {
                xercesc::DOMDocument *srcXmlDoc;
                const oscObjectBase *obj = member->getObject();
                if (obj)
                {
                    xercesc::DOMDocument *memSrcXmlDoc = obj->getSource()->getXmlDoc();
                    srcXmlDoc = memSrcXmlDoc;

                    if (srcXmlDoc != document)
                    {
                        document = srcXmlDoc;
                        currentElement = srcXmlDoc->getDocumentElement();
                    }

                    member->writeToDOM(currentElement,document);
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

    //find child with name INCLUDE
    bool inclPresent = false;
    xercesc::DOMNode *includeMember;

    for (int i = 0; i < membersList->getLength(); i++)
    {
        if (xercesc::XMLString::transcode(membersList->item(i)->getNodeName()) == INCLUDE)
        {
            includeMember = membersList->item(i);
            inclPresent = true;
        }
    }

    //if child with name INCLUDE present
    if (inclPresent)
    {
        xercesc::DOMNode *cloneOfMember = includeMember->cloneNode(true);
        currentElement->removeChild(includeMember);

        //set member with name INCLUDE to end of membersList
        currentElement->appendChild(cloneOfMember);

        //set member with name INLCUDE to the beginning of membersList
        //currentElement->insertBefore(cloneOfMember, currentElement->getFirstChild());

        // membersList is "live" so the new order is recognized automatically
    }


    //Attributes of current element
    //
    for (unsigned int attrIndex = 0; attrIndex < attributes->getLength(); ++attrIndex)
    {
        xercesc::DOMAttr *attribute = dynamic_cast<xercesc::DOMAttr *>(attributes->item(attrIndex));
        if(attribute !=NULL)
        {
            oscMember *m = members[xercesc::XMLString::transcode(attribute->getName())];
            if(m)
            {
                oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);
                if(am)
                {
                    std::cerr << "Array values not yet implemented" << std::endl;
                }
                else
                {
                    oscMemberValue::MemberTypes type = m->getType();
                    oscMemberValue *v = oscFactories::instance()->valueFactory->create(type);
                    if(v)
                    {
                        if(type == oscMemberValue::ENUM)
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
                        std::cerr << "could not create a value of type " << m->getType() << std::endl;
                    }
                }
            }
            else
            {
                std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member called " << xercesc::XMLString::transcode(attribute->getName()) << std::endl;
            }
        }
    }

    //children of current element
    //
    for (unsigned int memberIndex = 0; memberIndex < membersList->getLength(); ++memberIndex)
    {
        xercesc::DOMElement *memberElem = dynamic_cast<xercesc::DOMElement *>(membersList->item(memberIndex));
        if(memberElem !=NULL)
        {
            std::string memberName = xercesc::XMLString::transcode(memberElem->getNodeName());
            oscMember *m = members[memberName];
            if(m)
            {
                std::string memTypeName = m->getTypeName();

                oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);
                if(am)
                {
                    xercesc::DOMNodeList *arrayMembersList = memberElem->getChildNodes();
                    for (unsigned int arrayIndex = 0; arrayIndex < arrayMembersList->getLength(); ++arrayIndex)
                    {
                        xercesc::DOMElement *arrayMemElem = dynamic_cast<xercesc::DOMElement *>(arrayMembersList->item(arrayIndex));
                        std::string arrayMemName = xercesc::XMLString::transcode(arrayMemElem->getNodeName());
                        oscMember *ame = members[arrayMemName];
                        std::string arrayMemTypeName = ame->getTypeName();
                        oscObjectBase *obj = oscFactories::instance()->objectFactory->create(arrayMemTypeName);
                        if(obj)
                        {
                            obj->initialize(base, src);
                            am->push_back(obj);
                            obj->parseFromXML(arrayMemElem, src);
                        }
                        else
                        {
                            std::cerr << "could not create an object array of type " << arrayMemTypeName << std::endl;
                        }
                    }
                }
                else
                {
                    //member has a value (exists)
                    if ( m->exists() )
                    {
                        std::cerr << "\n Warning!" << std::endl;
                        std::cerr << "  Member \"" << m->getName() << "\" exists more than once as child of element \"" << xercesc::XMLString::transcode(currentElement->getNodeName()) << "\"" << std::endl;
                        std::cerr << "  Only first entry is used." << std::endl;
                        std::cerr << "  \"" << m->getName() << "\" from file: " << m->getOwner()->getSource()->getSrcFileName() << " is not used.\n" << std::endl;
                    }
                    //member has no value (doesn't exist)
                    else
                    {
                        oscObjectBase *obj = oscFactories::instance()->objectFactory->create(memTypeName);

                        if(obj)
                        {
                            obj->initialize(base, src);
                            m->setValue(obj);
                            obj->parseFromXML(memberElem, src);
                        }
                        else
                        {
                            std::cerr << "could not create an object member of type " << memTypeName << std::endl;
                        }
                    }
                }
            }
            else
            {
                std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member called " << memberName << std::endl;
            }

            //read include files
            //
            if (memberName == INCLUDE)
            {
                oscSourceFile *inclSource = new oscSourceFile();
                inclSource->initialize(base);

                //map of attributes of member include
                xercesc::DOMNamedNodeMap *inclMemAttrMap = memberElem->getAttributes();

                //filename to read from
                xercesc::DOMAttr *incMemAttr =
                        dynamic_cast<xercesc::DOMAttr *>(inclMemAttrMap->
                                getNamedItem(xercesc::XMLString::transcode("URL")));
                std::string srcFName = xercesc::XMLString::transcode(incMemAttr->getValue());

                //parent of member include
                xercesc::DOMElement *inclParent = dynamic_cast<xercesc::DOMElement *>(memberElem->getParentNode());
                std::string inclParentName = xercesc::XMLString::transcode(inclParent->getNodeName());

                //root element of file to include
                OpenScenarioBase *osbObj = new OpenScenarioBase();
                xercesc::DOMElement *srcFileRootElem = osbObj->getRootElement(srcFName);

                //check that file can be read
                if (srcFileRootElem != NULL)
                {
                    //root element of the file to include
                    std::string srcFileRootElemName = xercesc::XMLString::transcode(srcFileRootElem->getNodeName());

                    if (inclParentName == srcFileRootElemName)
                    {
                        inclSource->setVariables(srcFileRootElemName, srcFName);
                        this->base->addToSrcFileVec(inclSource);
                        this->parseFromXML(srcFileRootElem, inclSource);
                    }
                    else
                    {
                        std::cerr << "Include file root element doesn't match "
                                "the parent element of the include element." << std::endl;

                    }
                }
                else
                {
                    std::cerr << "\n Warning!" << std::endl;
                    std::cerr << "  Can't read include file  " << srcFName << "  to include elements under \"" << inclParentName << "\"!\n" << std::endl;
                }

                delete osbObj;
            }
        }
    }

    return true;
}
