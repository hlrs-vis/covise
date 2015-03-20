#ifndef COMUIELEMENTMANAGER_H
#define COMUIELEMENTMANAGER_H

#include <vector>
#include <iostream>
#include "../coMUIContainer.h"
#include "../coMUIWidget.h"
#include <boost/smart_ptr.hpp>


class coMUIElementManager
{
private:
    // variables, structures etc.
    struct entry;

    std::vector <entry> ElementList;

public:
    // constructor und destructor:
    coMUIElementManager();
    ~coMUIElementManager();

    // methods for access
    void addElement(std::string UniqueIdentifier, coMUIContainer* Parent);
    void addElement(std::string UniqueIdentifier, coMUIWidget* Widget);
    void removeElement(std::string UniqueIdentifier);
    coMUIContainer* getContainerByIdentifier(std::string UniqueIdentifier);
    coMUIWidget* getWidgetByIdentifier(std::string UniqueIdentifier);
    void deleteEntry(std::string UniqueiIdentifier);
    void printNames();
    bool isContainer(const std::string UniqueIdentifier);
};



#endif
