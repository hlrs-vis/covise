/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCOLORMAP_H
#define WSCOLORMAP_H

#include "WSExport.h"
#include "WSCoviseStub.h"

#include <QList>
#include <QString>

namespace covise
{

class WSLIBEXPORT WSColormapPin
{

public:
    WSColormapPin()
    {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
        a = 0.0f;
        position = 0.0f;
    }

    WSColormapPin(float red, float green, float blue, float alpha, float pos)
    {
        r = red;
        g = green;
        b = blue;
        a = alpha;
        position = pos;
    }

    WSColormapPin(const covise::covise__ColormapPin &pin)
    {
        r = pin.r;
        g = pin.g;
        b = pin.b;
        a = pin.a;
        position = pin.position;
    }
    virtual ~WSColormapPin(){};

    virtual covise::covise__ColormapPin getSerialisable() const;

    QString toString() const;

    float r;
    float g;
    float b;
    float a;
    float position;
};

class WSLIBEXPORT WSColormap
{
public:
    WSColormap();
    WSColormap(const QString &name, const QList<WSColormapPin> &pins);
    WSColormap(const covise::covise__Colormap &colormap);
    virtual ~WSColormap()
    {
    }

    const QList<WSColormapPin> &getPins() const
    {
        return this->pins;
    }

    virtual covise::covise__Colormap getSerialisable() const;

    void setFromSerialisable(const covise::covise__Colormap &serialisable);

    QString toString() const;
    QString toCoviseString() const;

private:
    QString name;
    QList<WSColormapPin> pins;
};
}
#endif // WSCOLORMAP_H
