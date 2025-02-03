#pragma once

#include "Element.h"
#include "Container.h"

namespace vive {
namespace ui {

//! semantic group of several UI \ref Element "elements"
class VIVE_UI_EXPORT Group: public Element, public Container {
 public:

     enum UpdateMask : UpdateMaskType
     {
         UpdateRelayout = 0x400,
     };
    Group(const std::string &name, Owner *owner);
    Group(Group *parent, const std::string &name);
    ~Group();

    //! add an Element to this Group
    bool add(Element *elem, int where=Append) override;
    //! remove an Element from this Group
    bool remove(Element *elem) override;

    //! request that graphical representation in all views is updated
    void update(UpdateMaskType updateMask = UpdateAll) const override;

    void allowRelayout(bool rl);
    bool allowRelayout() const { return m_allowRelayout; };

    virtual void save(covise::TokenBuffer& buf) const override;
    virtual void load(covise::TokenBuffer& buf) override;

protected:

    bool m_allowRelayout = false;
};

}
}
