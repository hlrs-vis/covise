#pragma once

#include "TextField.h"

#include <functional>

namespace vive {
namespace ui {

//! a graphical Element allowing for keyboard input

/** \note QLineEdit
    \note vvTUIEditField */
class VIVE_UI_EXPORT EditField: public TextField {

 public:
   EditField(Group *parent, const std::string &name);
   EditField(const std::string &name, Owner *owner);
   virtual ~EditField();

   using TextField::setValue;
   void setValue(double num);

   double number() const;

 protected:
};

}
}
