#include "EditField.h"
#include "Manager.h"
#include <net/tokenbuffer.h>

#include <sstream>
#include <cstdlib>

namespace opencover {
namespace ui {

EditField::EditField(Group *parent, const std::string &name)
: TextField(parent, name)
{
}

EditField::EditField(const std::string &name, Owner *owner)
: TextField(name, owner)
{
}

EditField::~EditField()
{
    manager()->remove(this);
}

void EditField::setValue(double num)
{
    std::stringstream str;
    str << num;

    TextField::setValue(str.str());
}

double EditField::number() const
{
    return atof(value().c_str());
}

}
}
