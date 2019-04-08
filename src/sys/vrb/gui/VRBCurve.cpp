/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


//#include "application.h"
#include "VRBCurve.h"
#include "VrbUiClientList.h"
#ifndef YAC
#include <covise/covise.h>
#endif

#include <QPainter>
#include <QWidget>
#include <QPainter>
#include <QRect>
#include <QTimer>
#include <QLabel>
//Added by qt3to4:
#include <QHideEvent>
#include <QShowEvent>
#include <QFrame>
#include <QPaintEngine>

#include <math.h>

#ifndef _WIN32
#include <unistd.h>
#endif

VRBCurve::VRBCurve(QWidget *parent)
    : QFrame(parent)
{
    setLineWidth(FrameWidth);
    setFrameStyle(Panel | Sunken);
    //setBackgroundMode( PaletteBase );
    setFixedSize(MaxSamples + 2 * FrameWidth, Range + 2 * FrameWidth);
    QPalette bgPalette;
    bgPalette.setColor(this->backgroundRole(), Qt::black);
    this->setPalette(bgPalette);

    QPalette fgPalette;
    fgPalette.setColor(this->foregroundRole(), Qt::green);
    this->setPalette(fgPalette);

    memset(yval, 0, sizeof(yval));
    timer = NULL;
    label = NULL;
    vrb = NULL;
    pos0 = 0;
    step = 5;
}

VRBCurve::~VRBCurve()
{
}

void VRBCurve::setClient(VrbUiClient *client)
{
    vrb = client;
}

void VRBCurve::setLabel(QLabel *text)
{
    label = text;
}

void VRBCurve::run()
{

    if (timer == NULL)
    {
        timer = new QTimer(this);
        connect(timer, SIGNAL(timeout()), this, SLOT(animate()));
    }

    timer->setSingleShot(false);
    timer->start(250);
}

void VRBCurve::stop()
{
    if (timer)
        timer->stop();
}

void VRBCurve::showEvent(QShowEvent *)
{
    run();
    repaint();
}

void VRBCurve::hideEvent(QHideEvent *)
{
    stop();
}

void VRBCurve::animate()
{
    QString s;
    int p, pval, y;
    //static int  def = 0;

    if (vrb == NULL)
        return;

    p = pos0;

    pval = vrb->getSentBPS() / 1000;
    y = pval;
    s.sprintf("%d KBit/s", pval);
    label->setText(s);

    //y    = (((Range-FrameWidth) * def)/100);
    //def++;
    //if(def == 100)
    //def = 0;

    for (int k = 0; k < step; k++)
    {
        yval[p] = y;
        p++;
        p %= MaxSamples;
    }

    scroll(-step, 0, QRect(FrameWidth, FrameWidth, MaxSamples, Range));
    pos0 = (pos0 + step) % MaxSamples;
}

void VRBCurve::paintEvent(QPaintEvent *p)
{
    //cerr << "drawContents " << endl;

    QRect r = p->rect();

    int vp = (r.left() - FrameWidth + pos0) % MaxSamples;

    /*cerr << "r1   " << r.left() << endl;
   cerr << "r2   " << r.right() << endl;
   cerr << "r3   " << r.bottom() << endl;
   cerr << "pos0 " << pos0 << endl;
   cerr << "vp   " << vp << endl;
   cerr << "=====" << endl;*/

    for (int x = r.left(); x <= r.right(); x++)
    {
        QPainter painter(this);
        painter.drawLine(x, r.bottom(), x, r.bottom() - yval[vp]);
        ++vp;
        vp %= MaxSamples;
    }
    QFrame::paintEvent(p);
}
