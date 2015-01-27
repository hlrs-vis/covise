/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigLong::coConfigLong(const QString &configGroupName, const QString &variable, const QString &section)
    : coConfigValue<long>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(const QString &variable, const QString &section)
    : coConfigValue<long>(variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(const QString &simpleVariable)
    : coConfigValue<long>(simpleVariable)
{

    update();
    //cerr << "coConfigLong::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(coConfigGroup *group, const QString &variable, const QString &section)
    : coConfigValue<long>(group, variable, section)
{

    update();
    //cerr << "coConfigLong::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigLong::coConfigLong(coConfigGroup *group, const QString &simpleVariable)
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

long coConfigLong::fromString(const QString &value) const
{
    QString v = value.toLower();
    long mult = 1;
    if (v.endsWith("k"))
        mult = 1024;
    else if (v.endsWith("m"))
        mult = 1024 * 1024;
    else if (v.endsWith("g"))
        mult = 1024 * 1024 * 1024;
    if (mult > 1)
        v = v.left(v.length() - 1);

    return v.toLong(0, 0) * mult;
}

QString coConfigLong::toString(const long &value) const
{
    return QString::number(value);
}

coConfigLong &coConfigLong::operator=(long value)
{
    //cerr << "coConfigLong::operator= info: setting to " << value << endl;
    coConfigValue<long>::operator=(value);
    return *this;
}
