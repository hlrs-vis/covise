/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSParameter.h"

#include <typeinfo>

std::map<QString, covise::WSParameter *> *covise::WSParameter::prototypes = 0;

covise::WSParameter::WSParameter(const QString &name, const QString &description, const QString &type)
    : QObject(0)
    , name(name)
    , type(type)
    , description(description)
    , enabled(true)
    , mapped(false)
{
    this->setObjectName(name);
}

covise::WSParameter::WSParameter(const covise::WSParameter &other)
    : QObject(0)
{
    this->name = other.name;
    this->type = other.type;
    this->description = other.description;
    this->enabled = other.enabled;
    this->mapped = other.mapped;
    this->setObjectName(this->name);
}

covise::WSParameter::~WSParameter()
{
}

const covise::covise__Parameter *covise::WSParameter::getSerialisable(covise::covise__Parameter *p) const
{
    p->name = getName().toStdString();
    p->type = getType().toStdString();
    p->description = getDescription().toStdString();
    p->mapped = this->mapped;
    return p;
}

covise::WSParameter *covise::WSParameter::create(covise::covise__Parameter *parameter)
{
    QString className = typeid(*parameter).name();
    if (covise::WSParameter::prototypes->find(className) != covise::WSParameter::prototypes->end())
    {
        covise::WSParameter *param = (*covise::WSParameter::prototypes)[className]->clone();
        param->setName(QString::fromStdString(parameter->name));
        param->setValueFromSerialisable(parameter);
        return param;
    }
    else
    {
        std::cerr << "WSParameter::create err: cannot find parameter for class " << qPrintable(className) << std::endl;
        for (std::map<QString, covise::WSParameter *>::iterator i = covise::WSParameter::prototypes->begin();
             i != covise::WSParameter::prototypes->end(); ++i)
            std::cerr << " " << qPrintable(i->first);
        std::cerr << std::endl;
        return 0;
    }
}

bool covise::WSParameter::equals(const covise::covise__Parameter *first, const covise::covise__Parameter *second)
{
    return (first->name == second->name) && (first->type == second->type) && (first->mapped == second->mapped) && (first->description == second->description);
}

int covise::WSParameter::getComponentCount() const
{
    return 1;
}

// EOF
