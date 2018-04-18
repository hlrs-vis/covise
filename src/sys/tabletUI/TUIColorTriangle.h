/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUICOLORTRIANGLE_H
#define TUICOLORTRIANGLE_H

#include <QObject>

#include "TUIElement.h"

class QtColorTriangle;
class QColor;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIColorTriangle : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIColorTriangle(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIColorTriangle();

    virtual void setValue(TabletValue type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual const char *getClassName() const;

public slots:

    void changeColor(const QColor &col);
    void releaseColor(const QColor &col);

protected:
    QtColorTriangle *colorTriangle;
    float red;
    float green;
    float blue;
    float alpha;
};
#endif
