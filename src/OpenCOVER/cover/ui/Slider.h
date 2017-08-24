#ifndef UI_SLIDER_H
#define UI_SLIDER_H

#include "Element.h"

#include <functional>
#include <limits>

namespace opencover {
namespace ui {

class COVER_UI_EXPORT Slider: public Element {
 public:
   typedef double ValueType;
   enum Presentation
   {
       AsSlider,
       AsDial
   };

   Slider(const std::string &name, Owner *owner);
   Slider(Group *parent, const std::string &name);

   void setMoving(bool flag);
   bool isMoving() const;

   ValueType value() const;
   void setValue(ValueType val);
   void setBounds(ValueType min, ValueType max);
   ValueType min() const;
   ValueType max() const;

   void setInteger(bool flag);
   bool integer() const;
   Presentation presentation() const;
   void setPresentation(Presentation pres);

    void setCallback(const std::function<void(ValueType, bool)> &f);
    std::function<void(ValueType, bool)> callback() const;

    void triggerImplementation() const override;
    void update() const override;

 private:
    bool m_integer = false;
    Presentation m_presentation = AsSlider;
    bool m_moving = false;
    ValueType m_value = ValueType(0);
    ValueType m_min = std::numeric_limits<ValueType>::lowest();
    ValueType m_max = std::numeric_limits<ValueType>::max();
    std::function<void(double, bool)> m_callback;
};

}
}
#endif
