/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_SLIDER_H
#define CO_TUI_SLIDER_H

#include <QObject>

#include "TUIElement.h"

class QLineEdit;
class QSlider;

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUISlider : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUISlider(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUISlider();
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);
    virtual void setValue(int type, covise::TokenBuffer &);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;
    void setPos(int x, int y);
    QLineEdit *string;
    QSlider *slider;
    int min;
    int max;
    int value;

public slots:

    void sliderChanged(int index);
    void pressed();
    void released();

protected:
};
#endif
