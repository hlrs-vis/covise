/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coTypes.h>

#ifdef YAC
#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#endif
#endif

#include "qslider.h"
#include "qpushbutton.h"
#include "qlabel.h"
#include "qlayout.h"
#include "qlineedit.h"
#include "qstring.h"
#include "qobject.h"
#include "qtimer.h"
#include "qlabel.h"
#include "qspinbox.h"
#include "qapplication.h"
//Added by qt3to4:
#include <QPixmap>
#include <QHBoxLayout>

#include <string.h>
#include <math.h>

#ifndef YAC
#include "InvCommunicator.h"
#endif

#include "InvSequencer.h"

#ifndef YAC
#include "InvObjectManager.h"
#else
#include "InvObjectManager_yac.h"
#endif

#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif

#include "XPM/1leftarrow.xpm"
#include "XPM/2leftarrow.xpm"
#include "XPM/1rightarrow.xpm"
#include "XPM/2rightarrow.xpm"
#include "XPM/stop.xpm"

//**************************************************************************
// CLASS InvSequencer
//**************************************************************************
InvSequencer::InvSequencer(QWidget *parent, const char *name)
    : QWidget(parent)

{
    setWindowTitle(name);
    min = 0;
    max = 0;
    val = 0;

// default sync mode
#ifdef YAC
    sync_flag = InvMain::SYNC_LOOSE;
#else
    sync_flag = InvMain::SYNC_TIGHT;
#endif

    // set timer
    maxFramesPerSecond = 24;
    nframes = 1000;
    animate = false;
    animationStarted = false;

    // make the layout
    QHBoxLayout *vb = new QHBoxLayout(this);
    makePlayer(vb);

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(allowNextAnimationStep()));

    //hide();
}

//------------------------------------------------------------------------
// make the Player layout
//------------------------------------------------------------------------
void InvSequencer::makePlayer(QHBoxLayout *box)
{
    int padX = 8;
    int padY = 4;

    // play back
    left2 = new QPushButton(QPixmap(two_leftarrow_xpm), "", this);
    connect(left2, SIGNAL(clicked()), this, SLOT(reverseCB()));
    left2->setFixedSize(20 + padX,
                        20 + padY);
    box->addWidget(left2);

    // one step back
    left1 = new QPushButton(QPixmap(one_leftarrow_xpm), "", this);
    left1->setFixedSize(20 + padX,
                        20 + padY);
    connect(left1, SIGNAL(clicked()), this, SLOT(left1CB()));
    box->addWidget(left1);

    // stop
    stop = new QPushButton(QPixmap(stop_xpm), "", this);
    stop->setFixedSize(20 + padX,
                       20 + padY);
    connect(stop, SIGNAL(clicked()), this, SLOT(stopCB()));
    box->addWidget(stop);

    // slider
    slider = new QSlider(Qt::Horizontal, this);
    slider->setTracking(true);
    //FIXME: slider->setTickmarks(QSlider::Below);
    slider->setTickInterval(1);
    slider->setPageStep(1);
    slider->setRange(min, max);
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(sliderCB(int)));
    box->addWidget(slider, 10);

    // one step forward
    right1 = new QPushButton(QPixmap(one_rightarrow_xpm), "", this);
    right1->setFixedSize(20 + padX,
                         20 + padY);
    connect(right1, SIGNAL(clicked()), this, SLOT(right1CB()));
    box->addWidget(right1);

    // play forward
    right2 = new QPushButton(QPixmap(two_rightarrow_xpm), "", this);
    right2->setFixedSize(20 + padX,
                         20 + padY);
    connect(right2, SIGNAL(clicked()), this, SLOT(playCB()));
    box->addWidget(right2);

    // current value
    timeStep = new QLabel("1/" + QString::number(max), this);
    box->addWidget(timeStep);

    // frames / sec
    fpsBox = new QSpinBox(this);
    fpsBox->setMaximum(MaxFPS);
    fpsBox->setMinimum(0);
    fpsBox->setSingleStep(1);
    fpsBox->setValue(maxFramesPerSecond);
    connect(fpsBox, SIGNAL(valueChanged(int)), this, SLOT(setFPS(int)));
    box->addWidget(fpsBox);

    box->addWidget(new QLabel("fps", this));

    setMinimumWidth(700);
    hide();
}

//======================================================================
InvSequencer::~InvSequencer()
//======================================================================
{
    // QTimers are deleted automatically with their parent
}

//======================================================================
void InvSequencer::sliderCB(int value)
//======================================================================
{
    char buf[255] = "";

    // do nothing if sane value
    if (val == value)
        return;

    // store current value
    val = value;

// send message for other renderer
#ifndef YAC
    sprintf(buf, "%d %d %d %d %d %d %d",
            slider->value(), min, max, min, max, seqState, 1);
    //fprintf(stderr, "InvSequencer::sliderCB: I should send a message, but I don't know how\n");
    renderer->cm->sendSequencerMessage(buf);
#else

    SoSwitch *swit = NULL;
    InvObjectManager::timestepSwitchList.reset();
    while ((swit = InvObjectManager::timestepSwitchList.next()) != NULL)
    {
        swit->whichChild.setValue(val + 1);
    }
#endif

    // update current shown value
    sprintf(buf, "%d", max);
    sprintf(buf, "%0*d", (int)strlen(buf), val);

    timeStep->setText(QString(buf) + "/" + QString::number(max));

    emit sequencerValueChanged(value);
}

//======================================================================
void InvSequencer::left1CB()
//======================================================================
{
    seqState = BACKWARD;
    animate = false;
    nextAnimationStep();
}

//======================================================================
void InvSequencer::stopCB()
//======================================================================
{
    seqState = STOP;
    animate = false;
}

//======================================================================
void InvSequencer::right1CB()
//======================================================================
{
    seqState = FORWARD;
    animate = false;
    nextAnimationStep();
}

//======================================================================
void InvSequencer::reverseCB()
//======================================================================
{
    seqState = PLAYBACKWARD;
    animate = true;
    animationLoop();
}

//======================================================================
void InvSequencer::playCB()
//======================================================================
{
    seqState = PLAYFORWARD;
    animate = true;
    animationLoop();
}

//======================================================================
void InvSequencer::setFPS(int fps)
{
    if (fps > MaxFPS)
    {
        maxFramesPerSecond = MaxFPS;
    }

    else if (fps <= 0)
    {
        maxFramesPerSecond = 0;
    }

    else
    {
        maxFramesPerSecond = fps;
    }

    if (animate)
    {
        if (maxFramesPerSecond > 0)
        {
            timer->start(1000 / maxFramesPerSecond);
        }
        else
        {
            timer->start(0);
        }
    }
}

//======================================================================
// set the sequencer active for user actions (master renderers)
//======================================================================
void InvSequencer::setSyncMode(QString message)
{

    if (message == "LOOSE")
        sync_flag = InvMain::SYNC_LOOSE;

    else if (message == "SYNC")
        sync_flag = InvMain::SYNC_SYNC;

    else
        sync_flag = InvMain::SYNC_TIGHT;
}

//======================================================================
// called by the list widget
//======================================================================
void InvSequencer::setSliderBounds(int minimum, int maximum, int value)
{

    val = value;
    min = minimum;
    max = maximum;

    slider->setRange(min, max);
    slider->setValue(val);

    //cerr << "value ______ " << val << endl;

    timeStep->setText(QString::number(val) + "/" + QString::number(max));
}

//======================================================================
// animation loop
//======================================================================
void InvSequencer::animationLoop()
{
    animate = true;

    if (maxFramesPerSecond > 0)
    {
        timer->start(1000 / maxFramesPerSecond);
    }
    else
    {
        timer->start(0);
    }
}

void InvSequencer::allowNextAnimationStep()
{
    nextAnimationStep();
}

void InvSequencer::nextAnimationStep()
{
    if (!animate)
    {
        timer->stop();
    }

    if (seqState == PLAYFORWARD || seqState == FORWARD)
    {
        if (slider->value() == slider->maximum())
        {
            slider->setValue(slider->minimum());
        }
        else
        {
            slider->setValue(slider->value() + 1);
        }
    }
    else if (seqState == PLAYBACKWARD || seqState == BACKWARD)
    {
        if (slider->value() == slider->minimum())
        {
            slider->setValue(slider->maximum());
        }
        else
        {
            slider->setValue(slider->value() - 1);
        }
    }
}
