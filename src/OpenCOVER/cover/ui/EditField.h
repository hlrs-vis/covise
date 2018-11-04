#ifndef UI_EDITFIELD_H
#define UI_EDITFIELD_H

#include "Element.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical Element allowing for keyboard input

/** \note QLineEdit
    \note coTUIEditField */
class COVER_UI_EXPORT EditField: public Element {

 public:
   enum UpdateMask: UpdateMaskType
   {
       UpdateValue = 0x100,
   };

   EditField(Group *parent, const std::string &name);
   EditField(const std::string &name, Owner *owner);
   virtual ~EditField();

   void setValue(const std::string &text);
   void setValue(double num);

   double number() const;
   std::string value() const;

   void setCallback(const std::function<void(const std::string &text)> &f);
   std::function<void(const std::string &)> callback() const;

   void triggerImplementation() const override;

    void update(UpdateMaskType mask) const override;

    void save(covise::TokenBuffer &buf) const override;
    void load(covise::TokenBuffer &buf) override;

 protected:
    std::function<void(const std::string &text)> m_callback;
    std::string m_value;
};

}
}
#endif
