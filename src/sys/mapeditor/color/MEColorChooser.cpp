/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QLabel>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMouseEvent>
#include <QPainter>
#include <qdrawutil.h>

#include "MEColorChooser.h"

static const int pWidth = 120;
static const int pHeight = 120;

/*!
   \class MEColorChooser
   \brief This class provides a color chooser widget used in the color map window
*/

/*****************************************************************************
 *
 * Class MEColorChooser
 *
 *****************************************************************************/

MEColorChooser::MEColorChooser(QWidget *parent)
    : QWidget(parent)
{
    // make a proper layout
    QHBoxLayout *topLay = new QHBoxLayout(this);

    QGroupBox *gb = new QGroupBox("Color", this);
    topLay->addWidget(gb);

    QGridLayout *grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(2);

    // set the color picker rectangle
    colorPicker = new MEColorPicker(gb);
    colorPicker->setFrameStyle(QFrame::Panel + QFrame::Sunken);
    colorPicker->setToolTip("Available colors");
    grid->addWidget(colorPicker, 0, 0);

    // set a kind of slider for luminance
    luminanceSlider = new MELuminanceSlider(gb);
    luminanceSlider->setFixedWidth(20);
    luminanceSlider->setToolTip("Luminance of current color");
    grid->addWidget(luminanceSlider, 0, 1);

    // add widgets for current color and hsv / rgb spin boxes
    colorValues = new MEColorValues(gb);
    grid->addWidget(colorValues, 0, 2);

    // add widgets that shows current color
    colorDisplay = new MEColorDisplay(gb);
    colorDisplay->setToolTip("Current color");
    colorDisplay->setMinimumWidth(60);
    grid->addWidget(colorDisplay, 0, 3);

    gb->setLayout(grid);

    gb = new QGroupBox("Opacity", this);
    topLay->addWidget(gb);
    grid = new QGridLayout();
    grid->setMargin(2);
    grid->setSpacing(1);

    // set a kind of slider for alpha value
    alphaSlider = new MEAlphaSlider(gb);
    alphaSlider->setFixedWidth(20);
    alphaSlider->setToolTip("This wigdet shows the opacity (alpha value) of the current color");
    grid->addWidget(alphaSlider, 0, 0);

    // add widgets for current alpha value
    alphaValue = new MEAlphaValue(gb);
    grid->addWidget(alphaValue, 0, 1);

    gb->setLayout(grid);

    connect(colorPicker, SIGNAL(HSVChanged(int, int)), this, SLOT(newHSV(int, int)));
    connect(colorPicker, SIGNAL(colorReleased()), this, SLOT(colorReleased()));
    connect(luminanceSlider, SIGNAL(HSVChanged(int, int, int)), this, SLOT(newHSV(int, int, int)));
    connect(luminanceSlider, SIGNAL(colorReleased()), this, SLOT(colorReleased()));
    connect(colorValues, SIGNAL(colorChanged(QColor)), this, SLOT(newColor(QColor)));
    connect(alphaSlider, SIGNAL(alphaChanged(int)), this, SLOT(newAlpha1(int)));
    connect(alphaSlider, SIGNAL(colorReleased()), this, SLOT(colorReleased()));
    connect(alphaValue, SIGNAL(alphaChanged(int)), this, SLOT(newAlpha2(int)));
}

//!
//! set the current color (from colormap)
//!
void MEColorChooser::setColor(const QColor &c)
{
    m_color = c;
    colorValues->setColor(c);
    colorDisplay->showColor(c);
    colorPicker->setHSV(c.hue(), c.saturation());
    luminanceSlider->setHSV(c.hue(), c.saturation(), c.value());
    alphaSlider->setAlpha(c.alpha());
    alphaValue->setAlpha(c.alpha());
}

//!
//! callback from luminanceSlider
//!
void MEColorChooser::newHSV(int h, int s, int v)
{
    int a = m_color.alpha();
    m_color.setHsv(h, s, v, a);
    colorPicker->setHSV(h, s);
    colorDisplay->showColor(m_color);
    colorValues->setColor(m_color);
    emit colorChanged(m_color);
}

//!
//! callback from colorPicker
//!
void MEColorChooser::newHSV(int h, int s)
{
    int v = m_color.value();
    int a = m_color.alpha();
    m_color.setHsv(h, s, v, a);
    luminanceSlider->setHSV(h, s);
    colorValues->setColor(m_color);
    colorDisplay->showColor(m_color);
    emit colorChanged(m_color);
}

//!
//! callback from alphaSlider
//!
void MEColorChooser::newAlpha1(int alpha)
{
    m_color.setAlpha(alpha);
    alphaValue->setAlpha(alpha);
    colorDisplay->showColor(m_color);
    emit colorChanged(m_color);
}

//!
//! callback from alphaValue
//!
void MEColorChooser::newAlpha2(int alpha)
{
    m_color.setAlpha(alpha);
    alphaSlider->setAlpha(alpha);
    colorDisplay->showColor(m_color);
    emit colorChanged(m_color);
}

//!
//! callback from color values
//!
void MEColorChooser::newColor(QColor c)
{
    int alpha = m_color.alpha();
    m_color = c;
    m_color.setAlpha(alpha);
    colorPicker->setHSV(c.hue(), c.saturation());
    luminanceSlider->setHSV(c.hue(), c.saturation(), c.value());
    colorDisplay->showColor(c);
    emit colorChanged(m_color);
}

//!
//! callback from all parts, a new color is available for the module
//!
void MEColorChooser::colorReleased()
{
    emit colorReady();
}

/*!
   \class MEColorPicker
   \brief This class provides a widget that allows to pick a color out of a color rectangle
*/

/*****************************************************************************
 *
 * Class coColorPicker
 *
 *****************************************************************************/
MEColorPicker::MEColorPicker(QWidget *parent)
    : QFrame(parent)
    , hue(0)
    , sat(0)
{
    setHSV(150, 255);

    QImage img(pHeight, pWidth, QImage::Format_RGB32);
    int x, y;
    for (y = 0; y < pHeight; y++)
        for (x = 0; x < pWidth; x++)
        {
            QPoint p(x, y);
            QColor c;
            c.setHsv(huePt(p), satPt(p), 200);
            img.setPixel(x, y, c.rgb());
        }

    pix = new QPixmap(QPixmap::fromImage(img));
    setAttribute(Qt::WA_NoSystemBackground);
    setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
}

MEColorPicker::~MEColorPicker()
{
    delete pix;
}

QPoint MEColorPicker::colPt()
{
    return QPoint((360 - hue) * (pWidth - 1) / 360, (255 - sat) * (pHeight - 1) / 255);
}

int MEColorPicker::huePt(const QPoint &pt)
{
    return 360 - pt.x() * 360 / (pHeight - 1);
}

int MEColorPicker::satPt(const QPoint &pt)
{
    return 255 - pt.y() * 255 / (pWidth - 1);
}

void MEColorPicker::setPoint(const QPoint &pt)
{
    setHSV(huePt(pt), satPt(pt));
}

QSize MEColorPicker::sizeHint() const
{
    return QSize(pHeight, pWidth);
}

QSizePolicy MEColorPicker::sizePolicy() const
{
    return QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void MEColorPicker::setHSV(int h, int s)
{
    int nhue = qMin(qMax(0, h), 359);
    int nsat = qMin(qMax(0, s), 255);
    if (nhue == hue && nsat == sat)
        return;
    QRect r(colPt(), QSize(20, 20));
    hue = nhue;
    sat = nsat;
    r = r.united(QRect(colPt(), QSize(20, 20)));
    r.translate(contentsRect().x() - 9, contentsRect().y() - 9);
    repaint(r);
}

void MEColorPicker::mouseMoveEvent(QMouseEvent *m)
{
    QPoint p = m->pos() - contentsRect().topLeft();
    setPoint(p);
    emit HSVChanged(hue, sat);
}

void MEColorPicker::mousePressEvent(QMouseEvent *m)
{
    QPoint p = m->pos() - contentsRect().topLeft();
    setPoint(p);
    emit HSVChanged(hue, sat);
}

void MEColorPicker::mouseReleaseEvent(QMouseEvent *)
{
    emit colorReleased();
}

void MEColorPicker::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    drawFrame(&p);
    QRect r = contentsRect();

    p.drawPixmap(r.topLeft(), *pix);
    QPoint pt = colPt() + r.topLeft();
    p.setPen(Qt::black);

    p.fillRect(pt.x() - 9, pt.y(), 20, 2, Qt::black);
    p.fillRect(pt.x(), pt.y() - 9, 2, 20, Qt::black);
}

/*!
   \class MELuminanceSlider
   \brief This class provides a widget for selecting the luminance of the current color
*/

/*****************************************************************************
 *
 * Class MELuminanceSlider
 *
 *****************************************************************************/
MELuminanceSlider::MELuminanceSlider(QWidget *parent)
    : QWidget(parent)
{
    hue = 100;
    val = 100;
    sat = 100;
    pix = 0;
    setFixedHeight(pHeight);
}

MELuminanceSlider::~MELuminanceSlider()
{
    delete pix;
}

int MELuminanceSlider::y2val(int y)
{
    int d = height() - 2 * coff - 1;
    return 255 - (y - coff) * 255 / d;
}

int MELuminanceSlider::val2y(int v)
{
    int d = height() - 2 * coff - 1;
    return coff + (255 - v) * d / 255;
}

void MELuminanceSlider::mouseMoveEvent(QMouseEvent *m)
{
    setVal(y2val(m->y()));
}

void MELuminanceSlider::mousePressEvent(QMouseEvent *m)
{
    setVal(y2val(m->y()));
}

void MELuminanceSlider::mouseReleaseEvent(QMouseEvent *)
{
    emit colorReleased();
}

void MELuminanceSlider::setVal(int v)
{
    if (val == v)
        return;
    val = qMax(0, qMin(v, 255));
    delete pix;
    pix = 0;
    repaint();
    emit HSVChanged(hue, sat, val);
}

//receives from a hue,sat chooser and relays.
void MELuminanceSlider::setHSV(int h, int s)
{
    setHSV(h, s, val);
}

void MELuminanceSlider::setHSV(int h, int s, int v)
{
    val = v;
    hue = h;
    sat = s;
    delete pix;
    pix = 0;
    update();
}

void MELuminanceSlider::paintEvent(QPaintEvent *)
{
    int w = width() - 5;

    QRect r(0, foff, w, height() - 2 * foff);
    int wi = r.width() - 2;
    int hi = r.height() - 2;
    if (!pix || pix->height() != hi || pix->width() != wi)
    {
        delete pix;
        QImage img(wi, hi, QImage::Format_RGB32);
        int y;
        for (y = 0; y < hi; y++)
        {
            QColor c;
            c.setHsv(hue, sat, y2val(y + coff));
            QRgb r = c.rgb();
            int x;
            for (x = 0; x < wi; x++)
                img.setPixel(x, y, r);
        }
        pix = new QPixmap(QPixmap::fromImage(img));
    }
    QPainter p(this);
    p.drawPixmap(1, coff, *pix);
    const QPalette &g = palette();
    qDrawShadePanel(&p, r, g, true);
    p.setPen(g.foreground().color());
    p.setBrush(g.foreground());
    QPolygon a;
    int y = val2y(val);
    a.setPoints(3, w, y, w + 5, y + 5, w + 5, y - 5);
    p.eraseRect(w, 0, 5, height());
    p.drawPolygon(a);
}

/*!
   \class MEColorDisplay
   \brief Class provides a widget displaying the current(used) color
*/

/*****************************************************************************
 *
 * Class MEColorDisplay
 *
 *****************************************************************************/
MEColorDisplay::MEColorDisplay(QWidget *parent)
    : QWidget(parent)
{
    pm_checker = QPixmap(":/icons/checker.xpm");
}

void MEColorDisplay::showColor(const QColor &c)
{
    col = c;
    alphacol = c;
    col.setAlpha(255);
    update();
}

void MEColorDisplay::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);
    QPainter p(this);

    // draw checker board
    int w2 = width() / 2;
    p.fillRect(0, 0, w2, height(), QBrush(col));
    p.fillRect(w2, 0, w2, height(), QBrush(pm_checker));
    p.fillRect(w2, 0, w2, height(), QBrush(alphacol));
}

/*!
   \class MEColorValues
   \brief Class provides a widget for displaying the HSV or RGB values of the current color
*/

/*****************************************************************************
 *
 * Class MEColorValues
 *
 *****************************************************************************/
MEColorValues::MEColorValues(QWidget *parent)
    : QWidget(parent)
{
    m_color.setRgb(0, 0, 0);

    QGridLayout *gl = new QGridLayout(this);

    hEd = new MEColorSpinBox(this);
    hEd->setRange(0, 359);
    hEd->setDecimals(0);
    hEd->setSingleStep(1);
    QLabel *l = new QLabel("Hu&e:", this);
    l->setBuddy(hEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 0, 0);
    gl->addWidget(hEd, 0, 1);

    sEd = new MEColorSpinBox(this);
    l = new QLabel("&Sat:", this);
    l->setBuddy(sEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 1, 0);
    gl->addWidget(sEd, 1, 1);

    vEd = new MEColorSpinBox(this);
    l = new QLabel("&Val:", this);
    l->setBuddy(vEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 2, 0);
    gl->addWidget(vEd, 2, 1);

    rEd = new MEColorSpinBox(this);
    l = new QLabel("&Red:", this);
    l->setBuddy(rEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 0, 2);
    gl->addWidget(rEd, 0, 3);

    gEd = new MEColorSpinBox(this);
    l = new QLabel("&Green:", this);
    l->setBuddy(gEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 1, 2);
    gl->addWidget(gEd, 1, 3);

    bEd = new MEColorSpinBox(this);
    l = new QLabel("Bl&ue:", this);
    l->setBuddy(bEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 2, 2);
    gl->addWidget(bEd, 2, 3);

    connect(hEd, SIGNAL(valueChanged(double)), this, SLOT(hsvEd(double)));
    connect(sEd, SIGNAL(valueChanged(double)), this, SLOT(hsvEd(double)));
    connect(vEd, SIGNAL(valueChanged(double)), this, SLOT(hsvEd(double)));

    connect(rEd, SIGNAL(valueChanged(double)), this, SLOT(rgbEd(double)));
    connect(gEd, SIGNAL(valueChanged(double)), this, SLOT(rgbEd(double)));
    connect(bEd, SIGNAL(valueChanged(double)), this, SLOT(rgbEd(double)));

    setFixedSize(this->sizeHint());
    setFixedHeight(pHeight);
}

void MEColorValues::rgbEd(double)
{
    int r = int(rEd->value() * 255.);
    int g = int(gEd->value() * 255.);
    int b = int(bEd->value() * 255.);

    m_color.setRgb(r, g, b);

    hEd->setValue(double(m_color.hue()));
    sEd->setValue(double(m_color.saturation() / 255));
    vEd->setValue(double(m_color.value() / 255));

    emit colorChanged(m_color);
}

void MEColorValues::hsvEd(double)
{
    int hue = int(hEd->value());
    int sat = int(sEd->value() * 255.);
    int val = int(vEd->value() * 255.);

    m_color.setHsv(hue, sat, val);

    rEd->setValue(double(m_color.redF()));
    gEd->setValue(double(m_color.greenF()));
    bEd->setValue(double(m_color.blueF()));

    emit colorChanged(m_color);
}

void MEColorValues::setColor(const QColor &c)
{
    m_color = c;
    hEd->setValue(double(c.hue()));
    sEd->setValue(double(c.saturationF()));
    vEd->setValue(double(c.valueF()));

    rEd->setValue(double(c.redF()));
    gEd->setValue(double(c.greenF()));
    bEd->setValue(double(c.blueF()));
}

/*!
   \class MEAlphaSlider
   \brief This class provides a widget for selecting a transparency value
*/

/*****************************************************************************
 *
 * Class MEAlphaSlider
 *
 *****************************************************************************/
MEAlphaSlider::MEAlphaSlider(QWidget *parent)
    : QWidget(parent)
{
    val = 255;
    pix = 0;
    setFixedHeight(pHeight);
}

MEAlphaSlider::~MEAlphaSlider()
{
    delete pix;
}

int MEAlphaSlider::y2val(int y)
{
    int d = height() - 2 * coff - 1;
    return 255 - (y - coff) * 255 / d;
}

int MEAlphaSlider::val2y(int v)
{
    int d = height() - 2 * coff - 1;
    return coff + (255 - v) * d / 255;
}

void MEAlphaSlider::mouseMoveEvent(QMouseEvent *m)
{
    setVal(y2val(m->y()));
}

void MEAlphaSlider::mousePressEvent(QMouseEvent *m)
{
    setVal(y2val(m->y()));
}

void MEAlphaSlider::mouseReleaseEvent(QMouseEvent *)
{
    emit colorReleased();
}

void MEAlphaSlider::setVal(int v)
{
    if (val == v)
        return;
    val = qMax(0, qMin(v, 255));
    delete pix;
    pix = 0;
    repaint();
    emit alphaChanged(val);
}

void MEAlphaSlider::setAlpha(int alpha)
{
    val = alpha;
    delete pix;
    pix = 0;
    repaint();
}

int MEAlphaSlider::getAlpha()
{
    return val;
}

void MEAlphaSlider::paintEvent(QPaintEvent *)
{
    int w = width() - 5;

    QRect r(0, foff, w, height() - 2 * foff);
    int wi = r.width() - 2;
    int hi = r.height() - 2;
    if (!pix || pix->height() != hi || pix->width() != wi)
    {
        delete pix;
        QImage img(wi, hi, QImage::Format_RGB32);
        int y;
        for (y = 0; y < hi; y++)
        {
            QColor c(y2val(y + coff), y2val(y + coff), y2val(y + coff));
            QRgb r = c.rgb();
            int x;
            for (x = 0; x < wi; x++)
                img.setPixel(x, y, r);
        }
        pix = new QPixmap(QPixmap::fromImage(img));
    }

    QPainter p(this);
    p.drawPixmap(1, coff, *pix);
    const QPalette &g = palette();
    qDrawShadePanel(&p, r, g, true);
    p.setPen(g.foreground().color());
    p.setBrush(g.foreground());
    QPolygon a;
    int y = val2y(val);
    a.setPoints(3, w, y, w + 5, y + 5, w + 5, y - 5);
    p.eraseRect(w, 0, 5, height());
    p.drawPolygon(a);
}

/*!
   \class MEAlphaValue
   \brief  This class provides a widget for displaying transparency value of the current color
*/

/*****************************************************************************
 *
 * Class MEAlphaValue
 *
 *****************************************************************************/
MEAlphaValue::MEAlphaValue(QWidget *parent)
    : QWidget(parent)
{
    QGridLayout *gl = new QGridLayout(this);

    aEd = new MEColorSpinBox(this);
    QLabel *l = new QLabel("Val:", this);
    l->setBuddy(aEd);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(l, 0, 0);
    gl->addWidget(aEd, 0, 1);

    connect(aEd, SIGNAL(valueChanged(double)), this, SLOT(alphaEd(double)));

    setFixedSize(this->sizeHint());
    setFixedHeight(pHeight);
}

void MEAlphaValue::alphaEd(double d)
{
    int alpha = int(d * 255.);
    emit alphaChanged(alpha);
}

void MEAlphaValue::setAlpha(int alpha)
{
    double value = double(alpha / 255.);
    aEd->setValue(value);
}

int MEAlphaValue::getAlpha()
{
    return int(aEd->value() * 255.);
}
