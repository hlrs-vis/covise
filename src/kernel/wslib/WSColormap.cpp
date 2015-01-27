/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSColormap.h"

#include "WSTools.h"

covise::covise__ColormapPin covise::WSColormapPin::getSerialisable() const
{
    covise::covise__ColormapPin pin;
    pin.r = this->r;
    pin.g = this->g;
    pin.b = this->b;
    pin.a = this->a;
    pin.position = this->position;
    return pin;
}

QString covise::WSColormapPin::toString() const
{
    return QString::number(r) + " " + QString::number(g) + " " + QString::number(b) + " " + QString::number(a) + " " + QString::number(position);
}

covise::WSColormap::WSColormap()
{
}

covise::WSColormap::WSColormap(const QString &name, const QList<covise::WSColormapPin> &pins)
    : name(covise::WSTools::fromCovise(name))
    , pins(pins)
{
}

covise::WSColormap::WSColormap(const covise::covise__Colormap &colormap)
{
    setFromSerialisable(colormap);
}

covise::covise__Colormap covise::WSColormap::getSerialisable() const
{
    covise::covise__Colormap colormap;
    colormap.name = this->name.toStdString();
    foreach (covise::WSColormapPin pin, this->pins)
    {
        colormap.pins.push_back(pin.getSerialisable());
    }
    return colormap;
}

void covise::WSColormap::setFromSerialisable(const covise::covise__Colormap &serialisable)
{
    this->name = QString::fromStdString(serialisable.name);
    this->pins.clear();
    for (std::vector<covise__ColormapPin>::const_iterator pin = serialisable.pins.begin();
         pin != serialisable.pins.end(); ++pin)
    {
        this->pins.push_back(covise::WSColormapPin(*pin));
    }
}

QString covise::WSColormap::toString() const
{
    QString rv = (this->name.contains(" ") ? "\"" + this->name + "\"" : this->name) + " " + QString::number(this->pins.size());

    foreach (covise::WSColormapPin pin, this->pins)
    {
        rv += " " + pin.toString();
    }
    return rv;
}

QString covise::WSColormap::toCoviseString() const
{
    QString rv = covise::WSTools::toCovise(this->name) + " " + QString::number(this->pins.size());
    foreach (covise::WSColormapPin pin, this->pins)
    {
        rv += " " + pin.toString();
    }
    return rv;
}
