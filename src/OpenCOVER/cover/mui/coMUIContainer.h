#ifndef COMUICONTAINER_H
#define COMUICONTAINER_H

#include "coMUIElement.h"
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coUIContainer.h>
#include <OpenVRUI/coMenu.h>
#include <iostream>
#include <boost/smart_ptr.hpp>


class COVEREXPORT coMUIContainer: public coMUIElement
{
public:
    // constructor:
    coMUIContainer();
    // destructor:
    ~coMUIContainer();

    // methods:
    virtual int getTUIID();
    virtual vrui::coMenu* getVRUI();
    virtual bool existTUI();                    // needs to be overwritten by inherited class
    virtual bool existVRUI();                   // needs to be overwritten by inherited class
    virtual void setPos(int posx, int posy)=0;    // needs to be overwritten by inherited class
    virtual opencover::coTUIElement* getTUI()=0;

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


#endif
