#ifndef UI_SPECIALELEMENT_H
#define UI_SPECIALELEMENT_H

#include "Element.h"
#include "View.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical Element acting as a template for view-specific code

class COVER_UI_EXPORT SpecialElement: public Element {

 public:
   SpecialElement(Group *parent, const std::string &name);
   SpecialElement(const std::string &name, Owner *owner);
   ~SpecialElement();

   void create(View::ViewElement *ve, View::ViewElement *parent);
   void destroy(View::ViewElement *ve);

   typedef std::function<void(SpecialElement *el, View::ViewElement *ve)> CreateFunc;
   typedef std::function<void(SpecialElement *el, View::ViewElement *ve)> DestroyFunc;

   void registerCreateDestroy(View::ViewType t, const CreateFunc &cf, const DestroyFunc &df);

protected:

   typedef std::pair<CreateFunc, DestroyFunc> CreateDestroy;
   std::map<View::ViewType, CreateDestroy> m_createDestroyFuncs;
};

}
}
#endif

