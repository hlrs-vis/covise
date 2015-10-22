/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscObjectBase.h>
#include <oscArrayMember.h>
#include <oscObjectVariable.h>
#include <OpenScenarioBase.h>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/util/XMLString.hpp>

#include <iostream>


using namespace OpenScenario;


oscObjectBase::oscObjectBase()
{
    base = NULL;
}
oscObjectBase::~oscObjectBase()
{
}

void oscObjectBase::initialize(OpenScenarioBase *b)
{
    base = b;
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
                member->writeToDOM(currentElement,document);
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
bool oscObjectBase::parseFromXML(xercesc::DOMElement *currentElement)
{
    xercesc::DOMNodeList *membersList = currentElement->getChildNodes();
    xercesc::DOMNamedNodeMap *attributes = currentElement->getAttributes();
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

    for (unsigned int memberIndex = 0; memberIndex < membersList->getLength(); ++memberIndex)
    {
        xercesc::DOMElement *memberElem = dynamic_cast<xercesc::DOMElement *>(membersList->item(memberIndex));
        if(memberElem !=NULL)
        {
            std::string memberName = xercesc::XMLString::transcode(memberElem->getNodeName());
            oscMember *m = members[memberName];
            std::string memTypeName = m->getTypeName();
            if(m)
            {
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
                            obj->initialize(base);
                            am->push_back(obj);
                            obj->parseFromXML(arrayMemElem);
                        }
                        else
                        {
                            std::cerr << "could not create an object array of type " << arrayMemTypeName << std::endl;
                        }
                    }
                }
                else
                {
                    oscObjectBase *obj = oscFactories::instance()->objectFactory->create(memTypeName);

                    if(obj)
                    {
                        obj->initialize(base);
                        m->setValue(obj);
                        obj->parseFromXML(memberElem);
                    }
                    else
                    {
                        std::cerr << "could not create an object member of type " << memTypeName << std::endl;
                    }
                }
            }
            else
            {
                std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member called " << memberName << std::endl;
            }
        }
    }

    return true;
}
