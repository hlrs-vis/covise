#include <vector>
#include <iostream>
#include "../Container.h"
#include "ElementManager.h"

using namespace mui;

// constructor
ElementManager::ElementManager()
{

}

// destructor
ElementManager::~ElementManager()
{

}

// creating a struct
struct ElementManager::entry
{
    entry();
    std::string name;
    mui::Element* pointer;
    bool container;
};
ElementManager::entry::entry()
{
    pointer = NULL;
}

// adds element to ElementList
void ElementManager::addElement(std::string uniqueIdentifier, mui::Element* element)
{
    bool exist=false;
    for (size_t i=0; i<elementList.size(); i++)
    {
        if (elementList[i].name==uniqueIdentifier)                     // name already exists
        {
            exist=true;
            std::cerr << "ERROR: ElementManager::addElement(): Element named " << elementList[i].name << " already exists. Choose another Name" << std::endl;
        }
    }
    if (!exist)                                                      // name doesn't exist yet -> new entry
    {
        elementList.push_back(entry());
        elementList[elementList.size()-1].name=uniqueIdentifier;
        elementList[elementList.size()-1].pointer = element;
        if (dynamic_cast<mui::Container *>(element))
        {
            elementList[elementList.size()-1].container = true;
        }
        else
        {
            elementList[elementList.size()-1].container = false;
        }
    }
};

// removes Element from ElementList
void ElementManager::removeElement(std::string uniqueIdentifier)
{
    for (size_t i=0; i<elementList.size(); i++)
    {
        if (elementList[i].name== uniqueIdentifier)
        {
            elementList.erase(elementList.begin()+i);
            --i;
        }
    }
}

// prints all names from ElementList to console
void ElementManager::printNames()
{

    std::cout << "Names in ElementManager: " << std::endl;
    for (size_t i=0; i< elementList.size(); ++i)
    {
        std::string toPrint = " ";
        toPrint.append(elementList[i].name);
        std::cout << toPrint << std::endl;
    }
    std::cout << std::endl;
}

// returns the container named "Identifier"
mui::Element* ElementManager::getElementByIdentifier(std::string uniqueIdentifier)
{
    for (size_t i=0; i<elementList.size(); i++)                           // go through all entrys
    {
        if (elementList[i].name == uniqueIdentifier)                       // match (name is equal)
        {
            return elementList[i].pointer;
        }
    }
    std::cerr << "ERROR: ElementManager::getParent(): Parent named " << uniqueIdentifier << " doesn't exist yet." << std::endl;
    return NULL;
}

// returns true, if element is a container; else returns false
bool ElementManager::isContainer(const std::string uniqueIdentifier)
{
    for (size_t i=0; i<elementList.size(); i++)
    {
        if (elementList[i].name == uniqueIdentifier)
        {
            return elementList[i].container;
        }
    }
    std::cerr << "ERROR: ElementManager::isContainer(): Element named " << uniqueIdentifier << " doesn't exist yet." << std::endl;
    return false;
}
