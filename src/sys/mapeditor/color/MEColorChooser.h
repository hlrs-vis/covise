/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_COLORCHOOSER_H
#define ME_COLORCHOOSER_H

#include <QFrame>
#include <QDoubleSpinBox>
#include <QDebug>

class MEColorSpinBox;
class MEColorDisplay;
class MEColorPicker;
class MEColorValues;
class MELuminanceSlider;
class MEAlphaSlider;
class MEAlphaValue;

//================================================
class MEColorChooser : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEColorChooser(QWidget *parent = 0);

private:
    QColor m_color;
    MEColorPicker *colorPicker;
    MELuminanceSlider *luminanceSlider;
    MEAlphaSlider *alphaSlider;
    MEColorValues *colorValues;
    MEAlphaValue *alphaValue;
    MEColorDisplay *colorDisplay;
    void showCustom(bool = true);

private slots:

    void newHSV(int h, int s, int v);
    void newHSV(int h, int s);
    void newAlpha1(int alpha);
    void newAlpha2(int alpha);
    void newColor(QColor c);
    void colorReleased();

public slots:

    void setColor(const QColor &);

signals:

    void colorChanged(const QColor &);
    void colorReady();
};

//================================================
class MEColorPicker : public QFrame
//================================================
{
    Q_OBJECT

public:
    MEColorPicker(QWidget *parent = 0);
    ~MEColorPicker();

private:
    int hue;
    int sat;
    QPixmap *pix;

    QPoint colPt();
    int huePt(const QPoint &pt);
    int satPt(const QPoint &pt);
    void setPoint(const QPoint &pt);

public slots:

    void setHSV(int h, int s);

signals:

    void HSVChanged(int h, int s);
    void colorReleased();

protected:
    QSize sizeHint() const;
    QSizePolicy sizePolicy() const;
    void paintEvent(QPaintEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
};

//================================================
class MELuminanceSlider : public QWidget
//================================================
{
    Q_OBJECT

public:
    MELuminanceSlider(QWidget *parent = 0);
    ~MELuminanceSlider();

private:
    enum //frame and contents offset
    {
        foff = 3,
        coff = 4
    };
    int val;
    int hue;
    int sat;
    QPixmap *pix;

    int y2val(int y);
    int val2y(int val);
    void setVal(int v);

public slots:

    void setHSV(int h, int s, int v);
    void setHSV(int h, int s);

signals:

    void HSVChanged(int h, int s, int v);
    void colorReleased();

protected:
    void paintEvent(QPaintEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
};

//================================================
class MEAlphaSlider : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEAlphaSlider(QWidget *parent = 0);
    ~MEAlphaSlider();

    int getAlpha();

private:
    enum //frame and contents offset
    {
        foff = 3,
        coff = 4
    };
    int val;
    QPixmap *pix;

    int y2val(int y);
    int val2y(int val);
    void setVal(int v);

public slots:

    void setAlpha(int alpha);

signals:

    void alphaChanged(int alpha);
    void colorReleased();

protected:
    void paintEvent(QPaintEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
};

//================================================
class MEColorDisplay : public QWidget
//================================================
{
public:
    MEColorDisplay(QWidget *parent = 0);

    void showColor(const QColor &c);

private:
    QColor col;
    QColor alphacol;
    QPixmap pm_checker;

protected:
    void paintEvent(QPaintEvent *);
};

//================================================
class MEColorValues : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEColorValues(QWidget *parent);

    void setNewHsv(int h, int s, int v);
    void setColor(const QColor &col);
    QColor currentColor() const
    {
        return m_color;
    }

private:
    MEColorSpinBox *hEd;
    MEColorSpinBox *sEd;
    MEColorSpinBox *vEd;
    MEColorSpinBox *rEd;
    MEColorSpinBox *gEd;
    MEColorSpinBox *bEd;

    QColor m_color;

signals:

    void colorChanged(QColor);

private slots:

    void rgbEd(double);
    void hsvEd(double);
};

//================================================
class MEAlphaValue : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEAlphaValue(QWidget *parent);

    void setAlpha(int alpha);
    int getAlpha();

private:
    MEColorSpinBox *aEd;

signals:

    void alphaChanged(int alpha);

private slots:

    void alphaEd(double);
};

//================================================
class MEColorSpinBox : public QDoubleSpinBox
//================================================
{
public:
    MEColorSpinBox(QWidget *parent)
        : QDoubleSpinBox(parent)
    {
        setRange(0, 1.0);
        setDecimals(3);
        setSingleStep(0.001);
    }

    void setValue(double v)
    {
        bool block = signalsBlocked();
        blockSignals(true);
        QDoubleSpinBox::setValue(v);
        blockSignals(block);
    }
};
#endif
