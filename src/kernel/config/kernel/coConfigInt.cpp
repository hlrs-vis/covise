/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigInt::coConfigInt(const QString &configGroupName, const QString &variable, const QString &section)
    : coConfigValue<int>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 0: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(const QString &variable, const QString &section)
    : coConfigValue<int>(variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(const QString &simpleVariable)
    : coConfigValue<int>(simpleVariable)
{

    update();
    //cerr << "coConfigInt::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(coConfigGroup *group, const QString &variable, const QString &section)
    : coConfigValue<int>(group, variable, section)
{

    update();
    //cerr << "coConfigInt::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigInt::coConfigInt(coConfigGroup *group, const QString &simpleVariable)
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

coConfigInt::~coConfigInt()
{
}

int coConfigInt::fromString(const QString &value) const
{
    QString v = value.toLower();
    int mult = 1;
    if (v.endsWith("k"))
        mult = 1024;
    else if (v.endsWith("m"))
        mult = 1024 * 1024;
    else if (v.endsWith("g"))
        mult = 1024 * 1024 * 1024;
    if (mult > 1)
        v = v.left(v.length() - 1);

    return v.toInt(0, 0) * mult;
}

QString coConfigInt::toString(const int &value) const
{
    return QString::number(value);
}

coConfigInt &coConfigInt::operator=(int value)
{
    //cerr << "coConfigInt::operator= info: setting to " << value << endl;
    coConfigValue<int>::operator=(value);
    return *this;
}
