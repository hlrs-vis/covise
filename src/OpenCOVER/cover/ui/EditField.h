#ifndef UI_EDITFIELD_H
#define UI_EDITFIELD_H

#include "TextField.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical Element allowing for keyboard input

/** \note QLineEdit
    \note coTUIEditField */
class COVER_UI_EXPORT EditField: public TextField {

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
#endif
