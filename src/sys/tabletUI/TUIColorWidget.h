/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_COLOR_WIDGET_H
#define CO_TUI_COLOR_WIDGET_H

#include <util/coTypes.h>
#include "TUIElement.h"
#include "TUILabel.h"
#include "qtcolortriangle.h"
#include <QObject>
#include <QGroupBox>
#include <QString>

class QSlider;
class QLineEdit;
class QString;
class QIntValidator;

class EditSlider : public QGroupBox
{
    Q_OBJECT
public:
    EditSlider(int min, int max, int step, int start, QWidget *parent, const QString &name);
    virtual ~EditSlider()
    {
    }
    virtual void setValue(int);

protected:
    QSlider *slider;
    QLineEdit *edit;
    QIntValidator *intValidator;
    QLabel *label;
    int val;

public slots:
    void moveSlot(int);
    void editChanged();
    void editTextChanged(const QString &);

signals:
    void moved(int);
};

class TUIColorWidget : public QFrame
{
    Q_OBJECT
public:
    TUIColorWidget(QWidget *parent);
    virtual ~TUIColorWidget();

    void setColor(const QColor &col, int a);
    QColor getColor()
    {
        return color;
    }

private:
    ColorTriangle *colorTriangle;
    EditSlider *redSlider;
    EditSlider *greenSlider;
    EditSlider *blueSlider;
    EditSlider *hueSlider;
    EditSlider *saturationSlider;
    EditSlider *valueSlider;
    //EditSlider		*alphaSlider;

    QColor color;

    void updateControls();

public slots:
    void changeTriangle(const QColor &col);
    void changeRed(int);
    void changeGreen(int);
    void changeBlue(int);
    void changeHue(int);
    void changeSaturation(int);
    void changeValue(int);
    void changeAlpha(int);

signals:
    void changedColor(const QColor &col);
    void changedAlpha(int);
};
#endif
