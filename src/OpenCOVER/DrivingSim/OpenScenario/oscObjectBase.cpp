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
    xercesc::DOMNodeList *childrenList = currentElement->getChildNodes();
    xercesc::DOMNamedNodeMap *attributes = currentElement->getAttributes();
    for (unsigned int childIndex = 0; childIndex < attributes->getLength(); ++childIndex)
    {
        xercesc::DOMAttr *attribute = dynamic_cast<xercesc::DOMAttr *>(attributes->item(childIndex));
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
    for (unsigned int childIndex = 0; childIndex < childrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *childElement = dynamic_cast<xercesc::DOMElement *>(childrenList->item(childIndex));
        if(childElement !=NULL)
        {
            std::string childName = xercesc::XMLString::transcode(childElement->getNodeName());
            oscMember *m = members[childName];
            if(m)
            {
                oscArrayMember *am = dynamic_cast<oscArrayMember *>(m);
                if(am)
                {
                    
                    xercesc::DOMNodeList *arrayChildrenList = childElement->getChildNodes();
                    for (unsigned int arrayIndex = 0; arrayIndex < arrayChildrenList->getLength(); ++arrayIndex)
                    {
                        
                        xercesc::DOMElement *arrayElement = dynamic_cast<xercesc::DOMElement *>(arrayChildrenList->item(arrayIndex));
                        std::string arrayElemName = xercesc::XMLString::transcode(arrayElement->getNodeName());
                        oscObjectBase *obj = oscFactories::instance()->objectFactory->create(arrayElemName);
                        if(obj)
                        {
                            obj->initialize(base);
                            am->push_back(obj);
                            obj->parseFromXML(arrayElement);
                        }
                        else
                        {
                            std::cerr << "could not create an object of type " << arrayElemName << std::endl; 
                        }
                    }
                }
                else
                {
                    oscObjectBase *obj = oscFactories::instance()->objectFactory->create(childName);
                    if(obj)
                    {
                        obj->initialize(base);
                        m->setValue(obj);
                        obj->parseFromXML(childElement);
                    }
                    else
                    {
                        std::cerr << "could not create an object of type " << childName << std::endl; 
                    }
                }
            }
            else
            {
                std::cerr << "Node " << xercesc::XMLString::transcode(currentElement->getNodeName()) << " does not have any member called " << childName << std::endl;
            }
        }
    }
    /*for(MemberMap::iterator it = members.begin();it != members.end();it++)
    {
        oscMember *member = it->second;
        if(member->type == OBJECT)
        {

            oscObjectBase *value = base->objectDactory.create(typeName);
            member->getValue(value);
            value.
            
        }
        else
        {
            oscMemberValue *value = oscMemberValue::factory. 
        }
    }*/
    return true;
}