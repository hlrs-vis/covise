/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
#include <util/string_util.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigBool::coConfigBool(const std::string &configGroupName, const std::string &variable, const std::string &section)
    : coConfigValue<bool>(configGroupName, variable, section)
{

    update();
    //std::cerr << "coConfigBool::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << std::endl;
}

coConfigBool::coConfigBool(const std::string &variable, const std::string &section)
    : coConfigValue<bool>(variable, section)
{

    update();
    //std::cerr << "coConfigBool::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << std::endl;
}

coConfigBool::coConfigBool(const std::string &simpleVariable)
    : coConfigValue<bool>(simpleVariable)
{

    update();
    //std::cerr << "coConfigBool::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << std::endl;
}

coConfigBool::coConfigBool(coConfigGroup *group, const std::string &variable, const std::string &section)
    : coConfigValue<bool>(group, variable, section)
{

    update();
    //std::cerr << "coConfigBool::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << std::endl;
}

coConfigBool::coConfigBool(coConfigGroup *group, const std::string &simpleVariable)
    : coConfigValue<bool>(group, simpleVariable)
{

    update();
    //std::cerr << "coConfigBool::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << std::endl;
}

coConfigBool::coConfigBool(const coConfigBool &value)
    : coConfigValue<bool>(value)
{
}

coConfigBool::~coConfigBool()
{
}

bool coConfigBool::fromString(const std::string &value) const
{
    return (toLower(value) == "on" || toLower(value) == "true" || atoi(value.c_str()) > 0);
}

std::string coConfigBool::toString(const bool &value) const
{
    return (value ? "on" : "off");
}

coConfigBool &coConfigBool::operator=(bool value)
{
    //std::cerr << "coConfigBool::operator= info: setting to " << value << std::endl;
    coConfigValue<bool>::operator=(value);
    return *this;
}
