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
class QLabel;

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
    virtual void setValue(int type, covise::TokenBuffer &) override;
    virtual void setLabel(QString textl) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;
    /// check if the Element or any ancestor is this classname
    void setPos(int x, int y) override;

public slots:
    void sliderChanged(int index);
    void pressed();
    void released();

protected:
    QLineEdit *string = nullptr;
    QSlider *slider = nullptr;
    QLabel *label = nullptr;
    int min;
    int max;
    int value;
};
#endif
