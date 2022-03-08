/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
#include <util/string_util.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigLong::coConfigLong(const std::string &configGroupName, const std::string &variable, const std::string &section)
    : coConfigValue<long>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(const std::string &variable, const std::string &section)
    : coConfigValue<long>(variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(const std::string &simpleVariable)
    : coConfigValue<long>(simpleVariable)
{

    update();
    //cerr << "coConfigLong::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(coConfigGroup *group, const std::string &variable, const std::string &section)
    : coConfigValue<long>(group, variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(coConfigGroup *group, const std::string &simpleVariable)
    : coConfigValue<long>(group, simpleVariable)
{

    update();
    //cerr << "coConfigLong::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(const coConfigLong &value)
    : coConfigValue<long>(value)
{
    //COCONFIGLOG("coConfigLong::<init> info: copy");
}

coConfigLong::~coConfigLong()
{
}

long coConfigLong::fromString(const std::string &value) const
{

    std::string v = toLower(value);
    int mult = 1;
    char end = v[v.size() - 1];
    if (end == 'k')
        mult = 1024;
    else if (end == 'm')
        mult = 1024 * 1024;
    else if (end == 'g')
        mult = 1024 * 1024 * 1024;
    if (mult > 1)
        v.pop_back();

    return std::stol(v) * mult;
}

std::string coConfigLong::toString(const long &value) const
{
    return std::to_string(value);
}

coConfigLong &coConfigLong::operator=(long value)
{
    //cerr << "coConfigLong::operator= info: setting to " << value << endl;
    coConfigValue<long>::operator=(value);
    return *this;
}
