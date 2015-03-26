#ifndef COMUIWIDGET_H
#define COMUIWIDGET_H

#include "coMUIElement.h"


class coMUIContainer;


class coMUIWidget: public coMUIElement
{
public:
    // methods:
    coMUIWidget();
    ~coMUIWidget();
    virtual opencover::coTUIElement* getTUI()=0;
    virtual void setPos(int,int)=0;
    virtual coMUIContainer *getParent()=0;
private:
    // methods:

    // variables:
};


#endif
