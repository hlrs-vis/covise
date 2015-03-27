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
    std::string name;
    Container* ContainerPointer;
    Widget* WidgetPointer;
    bool container;
};

// adds container-element to ElementList
void ElementManager::addElement(std::string UniqueIdentifier, Container* Container)
{
    bool exist=0;
    for (size_t i=0; i<ElementList.size(); i++)
    {
        if (ElementList[i].name==UniqueIdentifier)                     // name already exists
        {
            exist=1;
            std::cerr << "ERROR: ElementManager::addElement(): Element named " << ElementList[i].name << " already exists. Choose another Name" << std::endl;
        }
    }
    if (exist!=1)                                                      // name doesn't exist yet -> new entry
    {
        ElementList.push_back(entry());
        ElementList[ElementList.size()-1].name=UniqueIdentifier;
        ElementList[ElementList.size()-1].ContainerPointer = Container;
        ElementList[ElementList.size()-1].container = true;
    }
};

// adds widget-element to ElementList
void ElementManager::addElement(std::string UniqueIdentifier, Widget* Widget)
{
    bool exist=0;
    for (size_t i=0; i<ElementList.size(); i++)
    {
        if (ElementList[i].name==UniqueIdentifier)                     // name already exists
        {
            exist=1;
            std::cerr << "ERROR: ElementManager::addElement(): Element named " << ElementList[i].name << " already exists. Choose another Name" << std::endl;
        }
    }
    if (exist!=1)                                                      // name doesn't exist yet -> new entry
    {
        ElementList.push_back(entry());
        ElementList[ElementList.size()-1].name=UniqueIdentifier;
        ElementList[ElementList.size()-1].WidgetPointer = Widget;
        ElementList[ElementList.size()-1].container = false;
    }
};

// removes Element from ElementList
void ElementManager::removeElement(std::string UniqueIdentifier)
{
    for (size_t i=0; i<ElementList.size(); i++)
    {
        if (ElementList[i].name== UniqueIdentifier)
        {
            ElementList.erase(ElementList.begin()+i);
            --i;
        }
    }
}

// prints all names from ElementList to console
void ElementManager::printNames()
{

    std::cout << "Names in ElementManager: " << std::endl;
    for (size_t i=0; i< ElementList.size(); ++i)
    {
        std::string toPrint = " ";
        toPrint.append(ElementList[i].name);
        toPrint.append("; Parent: ");
        if (ElementList[i].container)
        {
            if (ElementList[i].ContainerPointer->getParent() != NULL)
            {
                toPrint.append(ElementList[i].ContainerPointer->getParent()->getUniqueIdentifier());
            }
        }
        else
        {
            toPrint.append(ElementList[i].WidgetPointer->getParent()->getUniqueIdentifier());
        }
        std::cout << toPrint << std::endl;
    }
    std::cout << std::endl;
}

// returns the container named "Identifier"
Container* ElementManager::getContainerByIdentifier(std::string UniqueIdentifier)
{
    for (size_t i=0; i<ElementList.size(); i++)                           // go through all entrys
    {
        if ((ElementList[i].name==UniqueIdentifier) && (ElementList[i].container == true))                       // match (name is equal)
        {
            return ElementList[i].ContainerPointer;
        }
    }
    std::cerr << "ERROR: ElementManager::getParent(): Parent named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return NULL;
}

// returns the widget named "Identifier"
Widget* ElementManager::getWidgetByIdentifier(std::string UniqueIdentifier)
{
    for (size_t i=0; i<ElementList.size(); i++)                // go through all entrys
    {
        if ((ElementList[i].name==UniqueIdentifier) && (ElementList[i].container == false))
        {                       // match (name is equal)
            return ElementList[i].WidgetPointer;
        }
    }
    std::cerr << "ERROR: ElementManager::getWidget(): Widget named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return NULL;
}

// returns true, if element is a container; else returns false
bool ElementManager::isContainer(const std::string UniqueIdentifier)
{
    for (size_t i=0; i<ElementList.size(); i++)
    {
        if (ElementList[i].name == UniqueIdentifier)
        {
            return ElementList[i].container;
        }
    }
    std::cerr << "ERROR: ElementManager::isContainer(): Element named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return false;
}


// delete entry with name "Name" from  ElementList
void ElementManager::deleteEntry(std::string UniqueIdentifier)
{
    for (size_t i=0; i<ElementList.size(); i++)
    {
        if (ElementList[i].name == UniqueIdentifier)
        {
            ElementList.erase(ElementList.begin()+i);
            i--;
        }
    }
    std::cerr << " ERROR: ElementManager::deleteEntry(): Parent named " << UniqueIdentifier << " doesn't exist yet. So it can't be deleted." << std::endl;
}
