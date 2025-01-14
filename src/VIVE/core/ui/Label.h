#pragma once

#include "Element.h"

namespace vive {
namespace ui {

//! a graphical Element showing a text label

/** \note QLabel
    \note vrui::coLabelMenuItem
    \note vvTUILabel
    */
class VIVE_UI_EXPORT Label: public Element {

 public:
   Label(Group *parent, const std::string &name);
   Label(const std::string &name, Owner *owner);
   ~Label();
};

}
}
