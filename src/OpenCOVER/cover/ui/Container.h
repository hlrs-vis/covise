#ifndef UI_CONTAINER_H
#define UI_CONTAINER_H

#include <memory>
#include <vector>

#include "Export.h"

namespace opencover {
namespace ui {

class Element;

//! mix-in class for containers of UI \ref Element "elements"
class COVER_UI_EXPORT Container {
 public:
    //! special arguments for where parameter fo add method
    enum Position {
        KeepFirst = -1,
        Front = KeepFirst,
        Append = -2,
        Back = Append,
        KeepLast = -3,
    };
    virtual ~Container();

    virtual bool add(Element *elem, int where=Append);
    virtual bool remove(Element *elem);

    size_t numChildren() const;
    Element *child(size_t index) const;
    int index(const Element *elem) const;

 protected:
    struct Child
    {
        Child(Element *elem, int where)
        : elem(elem), where(where)
        {}
        Element *elem = nullptr;
        int where = 0;
    };
    std::vector<Child> m_children;
    void clearChildren();
};

}
}
#endif
