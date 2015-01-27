/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigBool::coConfigBool(const QString &configGroupName, const QString &variable, const QString &section)
    : coConfigValue<bool>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigBool::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigBool::coConfigBool(const QString &variable, const QString &section)
    : coConfigValue<bool>(variable, section)
{

    update();
    //cerr << "coConfigBool::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigBool::coConfigBool(const QString &simpleVariable)
    : coConfigValue<bool>(simpleVariable)
{

    update();
    //cerr << "coConfigBool::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigBool::coConfigBool(coConfigGroup *group, const QString &variable, const QString &section)
    : coConfigValue<bool>(group, variable, section)
{

    update();
    //cerr << "coConfigBool::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigBool::coConfigBool(coConfigGroup *group, const QString &simpleVariable)
    : coConfigValue<bool>(group, simpleVariable)
{

    update();
    //cerr << "coConfigBool::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigBool::coConfigBool(const coConfigBool &value)
    : coConfigValue<bool>(value)
{
}

coConfigBool::~coConfigBool()
{
}

bool coConfigBool::fromString(const QString &value) const
{
    return (value.toLower() == "on" || value.toLower() == "true" || value.toInt() > 0);
}

QString coConfigBool::toString(const bool &value) const
{
    return (value ? "on" : "off");
}

coConfigBool &coConfigBool::operator=(bool value)
{
    //cerr << "coConfigBool::operator= info: setting to " << value << endl;
    coConfigValue<bool>::operator=(value);
    return *this;
}
