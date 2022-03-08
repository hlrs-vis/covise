/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#undef COCONFIGVALUE_USE_CACHE

#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigString::coConfigString(const std::string &configGroupName, const std::string &variable, const std::string &section)
    : coConfigValue<std::string>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const std::string &variable, const std::string &section)
    : coConfigValue<std::string>(variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const std::string &simpleVariable)
    : coConfigValue<std::string>(simpleVariable)
{

    update();
    //cerr << "coConfigString::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(coConfigGroup *group, const std::string &variable, const std::string &section)
    : coConfigValue<std::string>(group, variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(coConfigGroup *group, const std::string &simpleVariable)
    : coConfigValue<std::string>(group, simpleVariable)
{

    update();
    //cerr << "coConfigString::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const coConfigString &value)
    : coConfigValue<std::string>(value)
{
}

coConfigString::~coConfigString()
{
}

std::string coConfigString::fromString(const std::string &value) const
{
    return value;
}

std::string coConfigString::toString(const std::string &value) const
{
    return value;
}

coConfigString &coConfigString::operator=(std::string value)
{
    //cerr << "coConfigString::operator= info: setting to " << value << endl;
    coConfigValue<std::string>::operator=(value);
    return *this;
}
