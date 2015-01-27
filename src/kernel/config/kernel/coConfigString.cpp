/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#undef COCONFIGVALUE_USE_CACHE

#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigString::coConfigString(const QString &configGroupName, const QString &variable, const QString &section)
    : coConfigValue<QString>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const QString &variable, const QString &section)
    : coConfigValue<QString>(variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const QString &simpleVariable)
    : coConfigValue<QString>(simpleVariable)
{

    update();
    //cerr << "coConfigString::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(coConfigGroup *group, const QString &variable, const QString &section)
    : coConfigValue<QString>(group, variable, section)
{

    update();
    //cerr << "coConfigString::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(coConfigGroup *group, const QString &simpleVariable)
    : coConfigValue<QString>(group, simpleVariable)
{

    update();
    //cerr << "coConfigString::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigString::coConfigString(const coConfigString &value)
    : coConfigValue<QString>(value)
{
}

coConfigString::~coConfigString()
{
}

QString coConfigString::fromString(const QString &value) const
{
    return value;
}

QString coConfigString::toString(const QString &value) const
{
    return value;
}

coConfigString &coConfigString::operator=(QString value)
{
    //cerr << "coConfigString::operator= info: setting to " << value << endl;
    coConfigValue<QString>::operator=(value);
    return *this;
}
