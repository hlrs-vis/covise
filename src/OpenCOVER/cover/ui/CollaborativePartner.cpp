#include "CollaborativePartner.h"
#include "Group.h"
#include "ButtonGroup.h"
#include "Manager.h"

#include <net/tokenbuffer.h>

namespace opencover {
namespace ui {

CollaborativePartner::CollaborativePartner(const std::string &name, Owner *owner, ButtonGroup *bg, int id)
: Button(name, owner)
{
    setGroup(bg, id);
}

CollaborativePartner::CollaborativePartner(Group *parent, const std::string &name, ButtonGroup *bg, int id)
: Button(parent, name)
{
    setGroup(bg, id);
}

CollaborativePartner::~CollaborativePartner()
{
    manager()->remove(this);
    setGroup(nullptr, -1);
}

void CollaborativePartner::setViewpointCallback(const std::function<void (bool)> &f)
{
    m_viewpointCallback = f;
}

std::function<void (bool)> CollaborativePartner::viewpointCallback() const
{
    return m_viewpointCallback;
}

}
}
