/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_COLOR_TAB_H
#define CO_TUI_COLOR_TAB_H

#include <util/coTypes.h>
#include "TUITab.h"
#include "TUILabel.h"
#include "qtcolortriangle.h"
#include "TUIColorWidget.h"
#include <QObject>
#include <QGroupBox>

class QSlider;
class QLineEdit;
class QString;
class QIntValidator;

class TUIColorTab : public QObject, public TUITab
{
    Q_OBJECT
public:
    TUIColorTab(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIColorTab();
    virtual char *getClassName();
    virtual void setValue(int type, covise::TokenBuffer &tb);

private:
    QFrame *frame;
    QtColorTriangle *colorTriangle;
    EditSlider *redSlider;
    EditSlider *greenSlider;
    EditSlider *blueSlider;
    EditSlider *hueSlider;
    EditSlider *saturationSlider;
    EditSlider *valueSlider;
    EditSlider *alphaSlider;

    int red;
    int green;
    int blue;
    int alpha;
    int hue;
    int saturation;
    int value;

    void sendColor();
    void changeTriangleColor()
    {
        colorTriangle->setColor(QColor(red, green, blue));
    }
    void changeRedColor()
    {
        redSlider->setValue(red);
    }
    void changeGreenColor()
    {
        greenSlider->setValue(green);
    }
    void changeBlueColor()
    {
        blueSlider->setValue(blue);
        ;
    }
    void changeHueColor()
    {
        hueSlider->setValue(hue);
    }
    void changeSaturationColor()
    {
        saturationSlider->setValue(saturation);
    }
    void changeValueColor()
    {
        valueSlider->setValue(value);
    }

    void hSVtoRGB();
    void rGBtoHSV();

public slots:
    void changedTriangle(const QColor &col);
    void changedRed(int);
    void changedGreen(int);
    void changedBlue(int);
    void changedHue(int);
    void changedSaturation(int);
    void changedValue(int);
    void changedAlpha(int);
};
#endif
