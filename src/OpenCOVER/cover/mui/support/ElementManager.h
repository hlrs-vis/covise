#ifndef MUIELEMENTMANAGER_H
#define MUIELEMENTMANAGER_H

#include <vector>
#include <iostream>
#include "../Container.h"
#include "../Widget.h"
#include <boost/smart_ptr.hpp>

namespace mui
{
class ElementManager
{
private:
    // variables, structures etc.
    struct entry;

    std::vector <entry> ElementList;

public:
    // constructor und destructor:
    ElementManager();
    ~ElementManager();

    // methods for access
    void addElement(std::string UniqueIdentifier, Container* Parent);
    void addElement(std::string UniqueIdentifier, Widget* Widget);
    void removeElement(std::string UniqueIdentifier);
    Container* getContainerByIdentifier(std::string UniqueIdentifier);
    Widget* getWidgetByIdentifier(std::string UniqueIdentifier);
    void deleteEntry(std::string UniqueiIdentifier);
    void printNames();
    bool isContainer(const std::string UniqueIdentifier);
};
}


#endif
