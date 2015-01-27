/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSChoiceParameter.h>
#include <typeinfo>

covise::WSChoiceParameter *covise::WSChoiceParameter::prototype = new covise::WSChoiceParameter();

covise::WSChoiceParameter::WSChoiceParameter()
    : covise::WSParameter("", "", "Choice")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__ChoiceParameter).name(), this);
}

covise::WSChoiceParameter::WSChoiceParameter(const QString &name, const QString &description,
                                             const QStringList &inValue, int selected)
    : WSParameter(name, description, "Choice")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    setValue(inValue, selected);
}

covise::WSChoiceParameter::~WSChoiceParameter()
{
}

bool covise::WSChoiceParameter::setValue(const QStringList &inValue, int selected)
{

    this->parameter.choices.clear();
    foreach (QString v, inValue)
        this->parameter.choices.push_back(WSTools::fromCovise(v).toStdString());

    return setValue(selected, true);
}

bool covise::WSChoiceParameter::setValue(int selected)
{
    return setValue(selected, false);
}

bool covise::WSChoiceParameter::setValue(int selected, bool changed)
{

    changed = changed | (this->parameter.selected != selected);

    if (this->parameter.choices.size() > (unsigned int)selected)
        this->parameter.selected = selected;
    else
        this->parameter.selected = 1;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

QStringList covise::WSChoiceParameter::getValue() const
{
    QStringList rv;
    for (std::vector<std::string>::const_iterator v = this->parameter.choices.begin(); v != this->parameter.choices.end(); ++v)
        rv << QString::fromStdString(*v);
    return rv;
}

const QString covise::WSChoiceParameter::getSelectedValue()
{
    return QString::fromStdString(this->parameter.choices[this->parameter.selected]);
}

bool covise::WSChoiceParameter::setSelected(int index)
{

    bool changed = this->parameter.selected != index;

    if (this->parameter.choices.size() > (unsigned int)index)
        this->parameter.selected = index;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

int covise::WSChoiceParameter::getSelected() const
{
    return this->parameter.selected;
}

QString covise::WSChoiceParameter::toString() const
{
    QString rv = QString::number(this->parameter.selected);
    for (std::vector<std::string>::const_iterator v = this->parameter.choices.begin(); v != this->parameter.choices.end(); ++v)
    {
        rv += "|" + QString::fromStdString(*v);
    }
    //std::cerr << "WSChoiceParameter::toString info: " << qPrintable(this->getName()) << " = " << qPrintable(rv) << std::endl;
    return rv;
}

covise::WSParameter *covise::WSChoiceParameter::clone() const
{
    return new WSChoiceParameter(*this);
}

const covise::covise__Parameter *covise::WSChoiceParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSChoiceParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__ChoiceParameter *p = dynamic_cast<const covise::covise__ChoiceParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSChoiceParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
                  << " of type " << serialisable->type << std::endl;
        return false;
    }

    bool changed = !equals(&(this->parameter), p) || this->parameter.selected != p->selected || !std::equal(p->choices.begin(), p->choices.end(), this->parameter.choices.begin());

    if (changed)
    {
        this->parameter = *p;
        emit parameterChanged(this);
        return true;
    }
    else
    {
        return false;
    }
}

QString covise::WSChoiceParameter::toCoviseString() const
{
    QString rv = QString::number(this->parameter.selected + 1);
    for (std::vector<std::string>::const_iterator v = this->parameter.choices.begin(); v != this->parameter.choices.end(); ++v)
    {
        rv += " " + WSTools::toCovise(QString::fromStdString(*v));
    }
    rv = rv.trimmed();
    return rv;
}

int covise::WSChoiceParameter::getComponentCount() const
{
    return this->parameter.choices.size();
}
