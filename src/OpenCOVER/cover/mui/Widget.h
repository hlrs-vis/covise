#ifndef MUIWIDGET_H
#define MUIWIDGET_H

#include "Element.h"

namespace mui
{
class Container;


class Widget: public Element
{
public:
    // methods:
    Widget();
    ~Widget();
    virtual opencover::coTUIElement* getTUI()=0;
    virtual void setPos(int,int)=0;
    virtual Container *getParent()=0;
private:
    // methods:

    // variables:
};
} // end namespace

#endif
