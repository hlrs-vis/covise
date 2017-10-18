#ifndef UI_MENU_H
#define UI_MENU_H

#include "Group.h"

namespace opencover {
namespace ui {

//! a graphical item showing a menu grouping several interface elements together

/** \note QMenu */
class COVER_UI_EXPORT Menu: public Group {

 public:
   Menu(const std::string &name, Owner *owner);
   Menu(Group *parent, const std::string &name);
   ~Menu();

    virtual bool add(Element *elem) override;
    virtual bool remove(Element *elem) override;
};

}
}
#endif
