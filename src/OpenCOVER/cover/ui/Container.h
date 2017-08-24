#ifndef UI_CONTAINER_H
#define UI_CONTAINER_H

#include <memory>
#include <vector>

namespace opencover {
namespace ui {

class Element;

class Container {
 public:
    virtual ~Container();

    virtual bool add(Element *elem);
    virtual bool remove(Element *elem);

    size_t numChildren() const;
    Element *child(size_t index) const;

 protected:
    std::vector<Element *> m_children;
};

}
}
#endif
