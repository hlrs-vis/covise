#ifndef MUILISTENER_H
#define MUILISTENER_H

#include <cover/coVRPlugin.h>

namespace mui
{
class Element;

class COVEREXPORT Listener
{
public:
    Listener();
    ~Listener();
    virtual void muiEvent(Element *muiItem);
    virtual void muiPressEvent(Element *muiItem);
    virtual void muiValueChangeEvent(Element *muiItem);
    virtual void muiClickEvent(Element *muiItem);
    virtual void muiReleaseEvent(Element *muiItem);
};
} // end namespace

#endif // LISTENER_H
