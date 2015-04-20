#ifndef MUIELEMENTMANAGER_H
#define MUIELEMENTMANAGER_H

#include <vector>
#include <iostream>
#include "../Container.h"
#include <boost/smart_ptr.hpp>

namespace mui
{
class ElementManager
{
private:
    // variables, structures etc.
    struct entry;

    std::vector <entry> elementList;

public:
    // constructor und destructor:
    ElementManager();
    ~ElementManager();

    // methods for access
    void addElement(std::string uniqueIdentifier, mui::Element* element);
    void removeElement(std::string uniqueIdentifier);
    mui::Element* getElementByIdentifier(std::string uniqueIdentifier);
    void deleteEntry(std::string uniqueiIdentifier);
    void printNames();
    bool isContainer(const std::string uniqueIdentifier);
};
}


#endif
