/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLANEGROUP_H
#define PLANEGROUP_H

#include <QGroupBox>
#include "ui_PlaneGroup.h"
#include "ScreenPlane.h"
#include "Scene.h"

class MainWindow;

class PlaneGroup : public QGroupBox
{
    Q_OBJECT

public:
    PlaneGroup(MainWindow *parent = 0);
    ~PlaneGroup();
    void init();
    void update();
    void setupLineEdits();
    void setupLineEditsArray();
    void setupConnects();

public slots:
    //allgemein
    void lineEditEditingFinished(void);
    //LineEdits
    void edtWidthEditingFinished();
    void edtWidthResEditingFinished();
    void edtHeightEditingFinished();
    void edtHeightResEditingFinished();
    //ChkState
    void chkGeoMeshClicked(bool);

private:
    Ui::PlaneGroup ui;
    ScreenPlane *screen;
    MainWindow *mainWindow;
    QVector<QLineEdit *> lineEditsVec;
};

#endif // PLANEGROUP_H
