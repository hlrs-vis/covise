#ifndef UI_GROUP_H
#define UI_GROUP_H

#include "Element.h"
#include "Container.h"

namespace opencover {
namespace ui {

class COVER_UI_EXPORT Group: public Element, public Container {
 public:
    Group(const std::string &name, Owner *owner);
    Group(Group *parent, const std::string &name);

    bool add(Element *elem) override;
    bool remove(Element *elem) override;
};

}
}
#endif
