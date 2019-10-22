#ifndef UI_TEXTFIELD_H
#define UI_TEXTFIELD_H

#include "Element.h"

#include <functional>

namespace opencover {
namespace ui {

//! an Element allowing for text input

class COVER_UI_EXPORT TextField: public Element {

 public:
   enum UpdateMask: UpdateMaskType
   {
       UpdateValue = 0x100,
   };

   TextField(Group *parent, const std::string &name);
   TextField(const std::string &name, Owner *owner);
   virtual ~TextField();

   void setValue(const std::string &text);
   std::string value() const;

   void setCallback(const std::function<void(const std::string &text)> &f);
   std::function<void(const std::string &)> callback() const;

   void triggerImplementation() const override;

    void update(UpdateMaskType mask) const override;

    void save(covise::TokenBuffer &buf) const override;
    void load(covise::TokenBuffer &buf) override;

    void setShared(bool state) override;

 protected:
    std::function<void(const std::string &text)> m_callback;
    std::string m_value;

    typedef vrb::SharedState<std::string> SharedValue;
    void updateSharedState() override;
};

}
}
#endif
