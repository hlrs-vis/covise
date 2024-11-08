/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_FLOAT_SLIDER_H
#define CO_TUI_FLOAT_SLIDER_H

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
class TUIFloatSlider : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIFloatSlider(int id, int type, QWidget *w, int parent, QString namw);
    virtual ~TUIFloatSlider();
    virtual void setValue(TabletValue type, covise::TokenBuffer &) override;
    virtual void setLabel(QString textl) override;

    /// get the Element's classname
    virtual const char *getClassName() const override;
    void setPos(int x, int y) override;

public slots:
    void sliderChanged(int index);
    void pressed();
    void released();

protected:
    void showSliderValue(float min, float max, float val);

    QLineEdit *string = nullptr;
    QSlider *slider = nullptr;
    QLabel *label = nullptr;
    float min;
    float max;
    float value;
    int ival = 0;
    bool logScale = false;
};
#endif
