#include "SpecialElement.h"
#include "Manager.h"
#include "View.h"

namespace opencover {
namespace ui {

SpecialElement::SpecialElement(Group *parent, const std::string &name)
: Element(parent, name)
{
}

SpecialElement::SpecialElement(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

SpecialElement::~SpecialElement()
{
    manager()->remove(this);
}

void SpecialElement::create(View::ViewElement *ve, View::ViewElement *parent)
{
    auto t = ve->view->typeBit();
    auto it = m_createDestroyFuncs.find(t);
    if (it != m_createDestroyFuncs.end())
        it->second.first(this, ve);
}

void SpecialElement::destroy(View::ViewElement *ve)
{
    auto t = ve->view->typeBit();
    auto it = m_createDestroyFuncs.find(t);
    if (it != m_createDestroyFuncs.end())
        it->second.second(this, ve);
}

void SpecialElement::registerCreateDestroy(View::ViewType t, const CreateFunc &cf, const DestroyFunc &df)
{
    m_createDestroyFuncs[t] = std::make_pair(cf, df);
}

}
}

