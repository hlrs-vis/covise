/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_SEQUENCER_
#define _INV_SEQUENCER_

#include <util/coTypes.h>

#include <qwidget.h>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>

class Q3HBoxLayout;
class Q3VBoxLayout;
class QSlider;
class QPushButton;
class QSpinBox;
class QLabel;

//============================================================================
//
//  Class: InvSequencer
//
//  Sequencer for time frame stepping
//
//============================================================================
class InvSequencer : public QWidget
{
    Q_OBJECT

public:
    InvSequencer(QWidget *parent = 0, const char *name = 0);
    ~InvSequencer();

    enum
    {
        INACTIVE,
        ACTIVE
    };
    enum
    {
        STOP,
        FORWARD,
        BACKWARD,
        PLAYFORWARD,
        PLAYBACKWARD
    };

    void setSliderBounds(int min, int max, int val);
    int getValue()
    {
        return val;
    }
    int getMinimum()
    {
        return min;
    }
    int getMaximum()
    {
        return max;
    }

    static const int MaxFPS = 100;
    int maxFramesPerSecond;

    void setSyncMode(QString message);

    int getSeqState()
    {
        return seqState;
    };
    void setSeqState(const int &s)
    {
        seqState = s;
    };

    void stopPlayback(void)
    {
        seqState = STOP;
        animate = false;
    }

private:
    QSlider *slider;
    QPushButton *left1, *left2, *stop, *right1, *right2;
    QLabel *timeStep;
    QSpinBox *fpsBox;
    QTimer *timer;

    int sync_flag, seqState;
    int min, max, val; // slider state
    int nframes;
    bool animate;
    bool animationStarted;
    void nextAnimationStep();

    void animationLoop();
    void makePlayer(QHBoxLayout *);
    void updateSequencer();

signals:
    void sequencerValueChanged(int);

public slots:
    void playCB();
    void reverseCB();

private slots:
    void sliderCB(int);
    void left1CB();
    void stopCB();
    void right1CB();

    void setFPS(int);
    void allowNextAnimationStep();
};
#endif // _INV_SEQUENCER_
