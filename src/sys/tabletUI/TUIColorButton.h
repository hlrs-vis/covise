/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUIColorButton_H
#define TUIColorButton_H

#include <QObject>

#include "TUIElement.h"

class QtColorTriangle;
class QPushButton;
class QColor;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIColorButton : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIColorButton(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIColorButton();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);
    virtual void setValue(int type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

public slots:

    void changeColor(const QColor &col);
    void releaseColor(const QColor &col);
    void onColorButtonPressed();

protected:
    QPushButton *colorButton;
    float red;
    float green;
    float blue;
    float alpha;
};
#endif
