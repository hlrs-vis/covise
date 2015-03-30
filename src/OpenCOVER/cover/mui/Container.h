#ifndef MUICONTAINER_H
#define MUICONTAINER_H

#include "Element.h"
#include <iostream>
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coMenu;
class coRowMenu;
}

namespace mui
{
class COVEREXPORT Container: public Element
{
public:
    // constructor:
    Container();
    // destructor:
    ~Container();

    // methods:
    virtual int getTUIID();
    virtual vrui::coMenu* getVRUI();
    virtual bool existTUI();                    // needs to be overwritten by inherited class
    virtual bool existVRUI();                   // needs to be overwritten by inherited class
    virtual void setPos(int posx, int posy)=0;    // needs to be overwritten by inherited class
    virtual opencover::coTUIElement* getTUI()=0;
    virtual Container *getParent()=0;

private:

protected:

    // variables:
public:

private:
    int ID;
    bool visible_b;
    vrui::coRowMenu* menuItem;
protected:

};
} // end namespace

#endif
