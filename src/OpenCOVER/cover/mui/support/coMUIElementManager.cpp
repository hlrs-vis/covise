#include <vector>
#include <iostream>
#include "../coMUIContainer.h"
#include "coMUIElementManager.h"

// constructor
coMUIElementManager::coMUIElementManager(){

}

// destructor
coMUIElementManager::~coMUIElementManager(){

}

// creating a struct
struct coMUIElementManager::entry{
    std::string name;
    coMUIContainer* ContainerPointer;
    coMUIWidget* WidgetPointer;
    bool Container;
};

// adds container-element to ElementList
void coMUIElementManager::addElement(std::string UniqueIdentifier, coMUIContainer* Container){
    bool exist=0;
    for (int i=0; i<ElementList.size(); i++){
        if (ElementList[i].name==UniqueIdentifier){                     // name already exists
            exist=1;
            std::cerr << "ERROR: coMUIElementManager::addElement(): Element named " << ElementList[i].name << " already exists. Choose another Name" << std::endl;
        }
    }
    if (exist!=1){                                                      // name doesn't exist yet -> new entry
        ElementList.push_back(entry());
        ElementList[ElementList.size()-1].name=UniqueIdentifier;
        ElementList[ElementList.size()-1].ContainerPointer = Container;
        ElementList[ElementList.size()-1].Container = true;
    }
};

// adds widget-element to ElementList
void coMUIElementManager::addElement(std::string UniqueIdentifier, coMUIWidget* Widget){
    bool exist=0;
    for (int i=0; i<ElementList.size(); i++){
        if (ElementList[i].name==UniqueIdentifier){                     // name already exists
            exist=1;
            std::cerr << "ERROR: coMUIElementManager::addElement(): Element named " << ElementList[i].name << " already exists. Choose another Name" << std::endl;
        }
    }
    if (exist!=1){                                                      // name doesn't exist yet -> new entry
        ElementList.push_back(entry());
        ElementList[ElementList.size()-1].name=UniqueIdentifier;
        ElementList[ElementList.size()-1].WidgetPointer = Widget;
        ElementList[ElementList.size()-1].Container = false;
    }
};

// removes Element from ElementList
void coMUIElementManager::removeElement(std::string UniqueIdentifier){
    for (int i=0; i<ElementList.size(); i++){
        if (ElementList[i].name== UniqueIdentifier){
            ElementList.erase(ElementList.begin()+i);
            --i;
        }
    }
}

// prints all names from ElementList to console
void coMUIElementManager::printNames(){
    std::string toPrint = " ";
    for (int i=0; i< ElementList.size(); ++i){
        toPrint.append(ElementList[i].name);
        toPrint.append(", ");
    }
    std::cout << "Names in ElementManager: " << toPrint << std::endl;
}

// returns the container named "Identifier"
coMUIContainer* coMUIElementManager::getContainerByIdentifier(std::string UniqueIdentifier){
    for (int i=0; i<ElementList.size(); i++){                           // go through all entrys
        if ((ElementList[i].name==UniqueIdentifier) && (ElementList[i].Container == true)){                       // match (name is equal)
            return ElementList[i].ContainerPointer;
        }
    }
    std::cerr << "ERROR: coMUIElementManager::getParent(): Parent named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return NULL;
}

// returns the widget named "Identifier"
coMUIWidget* coMUIElementManager::getWidgetByIdentifier(std::string UniqueIdentifier){
    for (int i=0; i<ElementList.size(); i++){                // go through all entrys
        if ((ElementList[i].name==UniqueIdentifier) && (ElementList[i].Container == false)){                       // match (name is equal)
            return ElementList[i].WidgetPointer;
        }
    }
    std::cerr << "ERROR: coMUIElementManager::getWidget(): Widget named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return NULL;
}

// returns true, if element is a container; else returns false
bool coMUIElementManager::isContainer(const std::string UniqueIdentifier){
    for (int i=0; i<ElementList.size(); i++){
        if (ElementList[i].name == UniqueIdentifier){
            return ElementList[i].Container;
        }
    }
    std::cerr << "ERROR: coMUIElementManager::isContainer(): Element named " << UniqueIdentifier << " doesn't exist yet." << std::endl;
    return false;
}


// delete entry with name "Name" from  ElementList
void coMUIElementManager::deleteEntry(std::string UniqueIdentifier){
    for (int i=0; i<ElementList.size(); i++){
        if (ElementList[i].name == UniqueIdentifier){
            ElementList.erase(ElementList.begin()+i);
            i--;
        }
    }
    std::cerr << " ERROR: coMUIElementManager::deleteEntry(): Parent named " << UniqueIdentifier << " doesn't exist yet. So it can't be deleted." << std::endl;
}
