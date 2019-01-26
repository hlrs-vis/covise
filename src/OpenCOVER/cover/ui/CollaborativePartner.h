#ifndef UI_COLLABORATIVEPARTNER_H
#define UI_COLLABORATIVEPARTNER_H

#include "Button.h"

#include <functional>

namespace opencover {
namespace ui {

class ButtonGroup;

//! a graphical item showing a text label that can be pressed to toggle between two states

/** \note VruiPartnerMenuItem */
class COVER_UI_EXPORT CollaborativePartner: public Button {
    friend class ButtonGroup;
public:
    enum UpdateMask: UpdateMaskType
    {
        UpdateViewpoint = 0x1000,
    };
    CollaborativePartner(const std::string &name, Owner *owner, ButtonGroup *bg=nullptr, int id=0);
    CollaborativePartner(Group *parent, const std::string &name, ButtonGroup *bg=nullptr, int id=0);
    ~CollaborativePartner();

    void setViewpointCallback(const std::function<void(bool)> &f);
    std::function<void(bool)> viewpointCallback() const;

private:
    std::function<void(bool)> m_viewpointCallback;
};

}
}
#endif
