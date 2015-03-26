#ifndef COMUILISTENER_H
#define COMUILISTENER_H

#include <cover/coVRPlugin.h>

class coMUIElement;

class COVEREXPORT coMUIListener
{
public:
    coMUIListener();
    ~coMUIListener();
    virtual void muiEvent(coMUIElement *muiItem);
    virtual void muiPressEvent(coMUIElement *muiItem);
    virtual void muiValueChangeEvent(coMUIElement *muiItem);
    virtual void muiClickEvent(coMUIElement *muiItem);
    virtual void muiReleaseEvent(coMUIElement *muiItem);
};

#endif // COMUILISTENER_H
