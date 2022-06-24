/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
#include <util/string_util.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigInt::coConfigInt(const std::string &configGroupName, const std::string &variable, const std::string &section)
    : coConfigValue<int>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(const std::string &variable, const std::string &section)
    : coConfigValue<int>(variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(const std::string &simpleVariable)
    : coConfigValue<int>(simpleVariable)
{

    update();
    //cerr << "coConfigInt::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(coConfigGroup *group, const std::string &variable, const std::string &section)
    : coConfigValue<int>(group, variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(coConfigGroup *group, const std::string &simpleVariable)
    : coConfigValue<int>(group, simpleVariable)
{

    update();
    //cerr << "coConfigInt::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(const coConfigInt &value)
    : coConfigValue<int>(value)
{
    //COCONFIGLOG("coConfigInt::<init> info: copy");
}

int coConfigInt::fromString(const std::string &value) const
{
    if (value.size() == 0)
        return 0;
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

    return atoi(v.c_str()) * mult;
}

std::string coConfigInt::toString(const int &value) const
{
    return std::to_string(value);
}

coConfigInt &coConfigInt::operator=(int value)
{
    //cerr << "coConfigInt::operator= info: setting to " << value << endl;
    coConfigValue<int>::operator=(value);
    return *this;
}
