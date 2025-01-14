#pragma once

#include "Group.h"

namespace vive {
namespace ui {

//! a graphical item showing a menu grouping several interface elements together

/** \note QMenu */
class VIVE_UI_EXPORT Menu: public Group {

 public:
   Menu(const std::string &name, Owner *owner);
   Menu(Group *parent, const std::string &name);
   ~Menu();
};

}
}
