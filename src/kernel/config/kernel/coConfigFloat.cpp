/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COCONFIGVALUE_USE_CACHE
#include <config/coConfig.h>
//#include "coConfigValue.inl"

using namespace covise;

coConfigFloat::coConfigFloat(const QString &configGroupName, const QString &variable, const QString &section)
    : coConfigValue<float>(configGroupName, variable, section)
{

    update();
    //cerr << "coConfigFloat::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigFloat::coConfigFloat(const QString &variable, const QString &section)
    : coConfigValue<float>(variable, section)
{

    update();
    //cerr << "coConfigFloat::<init> info: 1: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigFloat::coConfigFloat(const QString &simpleVariable)
    : coConfigValue<float>(simpleVariable)
{

    update();
    //cerr << "coConfigFloat::<init> info: 2: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigFloat::coConfigFloat(coConfigGroup *group, const QString &variable, const QString &section)
    : coConfigValue<float>(group, variable, section)
{

    update();
    //cerr << "coConfigFloat::<init> info: 3: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigFloat::coConfigFloat(coConfigGroup *group, const QString &simpleVariable)
    : coConfigValue<float>(group, simpleVariable)
{

    update();
    //cerr << "coConfigFloat::<init> info: 4: " << this->variable << " " << this->section << " " << this->value << endl;
}

coConfigFloat::coConfigFloat(const coConfigFloat &value)
    : coConfigValue<float>(value)
{
    //cerr << "coConfigFloat::<init> info: copy" << endl;
}

coConfigFloat::~coConfigFloat()
{
}

float coConfigFloat::fromString(const QString &value) const
{
    return value.toFloat();
}

QString coConfigFloat::toString(const float &value) const
{
    return QString::number(value);
}

coConfigFloat &coConfigFloat::operator=(float value)
{
    //cerr << "coConfigFloat::operator= info: setting to " << value << endl;
    coConfigValue<float>::operator=(value);
    return *this;
}
