#ifndef UI_SLIDER_H
#define UI_SLIDER_H

#include "Element.h"

#include <functional>
#include <limits>

namespace opencover {
namespace ui {

//! a graphical item showing a text label and value that can be manipulated with a slider or dial

/** \note QSlider
    \note vrui::coSliderMenuItem, vrui::coPotiMenuItem
    \note coTUISlider, coTUIFloatSlider
    */
class COVER_UI_EXPORT Slider: public Element {
 public:
   typedef double ValueType;
   enum Presentation
   {
       AsSlider,
       AsDial
   };

   enum Scale
   {
       Linear,
       Logarithmic
   };
   enum UpdateMask: UpdateMaskType
   {
       UpdateValue = 0x100,
       UpdateScale = 0x200,
       UpdateIntegral = 0x400,
       UpdateBounds = 0x800,
   };

   Slider(const std::string &name, Owner *owner);
   Slider(Group *parent, const std::string &name);
   ~Slider();

   //! notify that slider is currently being manipulated
   void setMoving(bool flag);
   //! query whether slider is currently being manipulated
   bool isMoving() const;

   //! returns current slider value
   ValueType value() const;
   //! set slider value
   void setValue(ValueType val);
   //! set linearized slider value
   void setLinValue(ValueType val);
   //! set minimum and maximum of slider sange
   void setBounds(ValueType min, ValueType max);
   //! lower slider bound
   ValueType min() const;
   //! upper slider bound
   ValueType max() const;
   //! linearized lower slider bound
   ValueType linMin() const;
   //! linearized upper slider bound
   ValueType linMax() const;
   //! linearized slider value
   ValueType linValue() const;

   //! switch slider to representang just integral values
   void setIntegral(bool flag);
   //! whether slider is restricted to integral values
   bool integral() const;
   //! desired representation of slider (e.g. as a dial or as a slider)
   Presentation presentation() const;
   //! set desired representation of slider
   void setPresentation(Presentation pres);
   //! set desired scale (e.g. linear or logarithmic) of slider
   void setScale(Scale scale);
   //! retrieve scale of slider (e.g. linear or logarithmic)
   Scale scale() const;

   //! set function to be called whenever slider is manipulated, bool parameter is true if slider manipulation stops
    void setCallback(const std::function<void(ValueType, bool)> &f);
    std::function<void(ValueType, bool)> callback() const;

    void triggerImplementation() const override;

    void update(UpdateMaskType mask) const override;

    void save(covise::TokenBuffer &buf) const override;
    void load(covise::TokenBuffer &buf) override;

 private:
    bool m_integral = false;
    Presentation m_presentation = AsSlider;
    Scale m_scale = Linear;
    bool m_moving = false;
    ValueType m_value = ValueType(0);
    ValueType m_min = std::numeric_limits<ValueType>::lowest();
    ValueType m_max = std::numeric_limits<ValueType>::max();
    std::function<void(double, bool)> m_callback;
};

}
}
#endif
