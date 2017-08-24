#ifndef UI_MENU_H
#define UI_MENU_H

#include "Group.h"

namespace opencover {
namespace ui {

class COVER_UI_EXPORT Menu: public Group {

 public:
   Menu(const std::string &name, Owner *owner);
   Menu(Group *parent, const std::string &name);

    virtual bool add(Element *elem) override;
    virtual bool remove(Element *elem) override;
};

}
}
#endif
