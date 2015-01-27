/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSColormapChoiceParameter.h>
#include <typeinfo>

covise::WSColormapChoiceParameter *
    covise::WSColormapChoiceParameter::prototype = new covise::WSColormapChoiceParameter();

covise::WSColormapChoiceParameter::WSColormapChoiceParameter()
    : covise::WSParameter("", "", "ColormapChoice")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__ColormapChoiceParameter).name(), this);
}

covise::WSColormapChoiceParameter::WSColormapChoiceParameter(const QString &name,
                                                             const QString &description,
                                                             const QList<covise::WSColormap> &inValue,
                                                             int selected)
    : covise::WSParameter(name, description, "ColormapChoice")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    setValue(inValue, selected);
}

covise::WSColormapChoiceParameter::~WSColormapChoiceParameter()
{
}

bool covise::WSColormapChoiceParameter::setValue(const QList<WSColormap> &inValue, int selected)
{

    this->parameter.colormaps.clear();
    foreach (WSColormap v, inValue)
        this->parameter.colormaps.push_back(v.getSerialisable());

    return setValue(selected, true);
}

bool covise::WSColormapChoiceParameter::setValue(int selected)
{
    return setValue(selected, false);
}

bool covise::WSColormapChoiceParameter::setValue(int selected, bool changed)
{

    changed = changed | (this->parameter.selected != selected);

    if (this->parameter.colormaps.size() > (unsigned int)selected)
        this->parameter.selected = selected;
    else
        this->parameter.selected = 1;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

QList<covise::WSColormap> covise::WSColormapChoiceParameter::getValue() const
{
    QList<covise::WSColormap> rv;
    for (std::vector<covise::covise__Colormap>::const_iterator v = this->parameter.colormaps.begin();
         v != this->parameter.colormaps.end(); ++v)
        rv << covise::WSColormap(*v);
    return rv;
}

const covise::WSColormap covise::WSColormapChoiceParameter::getSelectedValue()
{
    return covise::WSColormap(this->parameter.colormaps[this->parameter.selected]);
}

bool covise::WSColormapChoiceParameter::setSelected(int index)
{

    bool changed = this->parameter.selected != index;

    if (this->parameter.colormaps.size() > (unsigned int)index)
        this->parameter.selected = index;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

int covise::WSColormapChoiceParameter::getSelected() const
{
    return this->parameter.selected;
}

QString covise::WSColormapChoiceParameter::toString() const
{
    QList<covise::WSColormap> colormaps = getValue();
    QString rv = QString::number(this->parameter.selected);
    rv += " " + QString::number(this->parameter.colormaps.size());
    foreach (covise::WSColormap colormap, colormaps)
    {
        rv += "|" + colormap.toString();
    }
    return rv;
}

covise::WSParameter *covise::WSColormapChoiceParameter::clone() const
{
    return new covise::WSColormapChoiceParameter(*this);
}

const covise::covise__Parameter *covise::WSColormapChoiceParameter::getSerialisable()
{
    return covise::WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSColormapChoiceParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__ColormapChoiceParameter *p = dynamic_cast<const covise::covise__ColormapChoiceParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSColormapChoiceParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
                  << " of type " << serialisable->type << std::endl;
        return false;
    }

    bool changed = !equals(&(this->parameter), p) || this->parameter.selected != p->selected || !std::equal(p->colormaps.begin(), p->colormaps.end(), this->parameter.colormaps.begin());

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

QString covise::WSColormapChoiceParameter::toCoviseString() const
{
    QList<covise::WSColormap> colormaps = getValue();
    QString rv = QString::number(this->parameter.selected);
    rv += " " + QString::number(this->parameter.colormaps.size());
    foreach (covise::WSColormap colormap, colormaps)
    {
        rv += " " + colormap.toCoviseString();
    }
    return rv.trimmed();
}

int covise::WSColormapChoiceParameter::getComponentCount() const
{
    return this->parameter.colormaps.size();
}
