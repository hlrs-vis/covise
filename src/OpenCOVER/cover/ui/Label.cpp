#include "Label.h"
#include "Manager.h"

namespace opencover {
namespace ui {

Label::Label(Group *parent, const std::string &name)
: Element(parent, name)
{
}

Label::Label(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

Label::~Label()
{
    manager()->remove(this);
}

}
}
