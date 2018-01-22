#ifndef UI_INPUT_H
#define UI_INPUT_H

#include "Element.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical Element allowing for keyboard input

class COVER_UI_EXPORT Input: public Element {

 public:
   enum UpdateMask: UpdateMaskType
   {
       UpdateValue = 0x100,
   };

   Input(Group *parent, const std::string &name);
   Input(const std::string &name, Owner *owner);
   virtual ~Input();

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
