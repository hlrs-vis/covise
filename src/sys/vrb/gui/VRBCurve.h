/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBCURVE_H
#define VRBCURVE_H

#include <QFrame>
//Added by qt3to4:
#include <QHideEvent>
#include <QLabel>
#include <QShowEvent>

class QPainter;
class QTimer;
class QLabel;
class VrbUiClient;
class VRBCurve : public QFrame
{
    Q_OBJECT
    enum
    {
        MaxSamples = 300,
        Range = 20,
        FrameWidth = 3
    };

public:
    VRBCurve(QWidget *parent = 0);
    ~VRBCurve();

    void run();
    void stop();
    void setClient(VrbUiClient *);
    void setLabel(QLabel *);

protected:
    virtual void showEvent(QShowEvent *);
    virtual void hideEvent(QHideEvent *);

public slots:
    void animate();

protected:
    virtual void paintEvent(QPaintEvent *);

private:
    int yval[MaxSamples];
    int pos0; // buffer pointer for x == 0
    int step;

    VrbUiClient *vrb;
    QTimer *timer;
    QLabel *label;
};
#endif
