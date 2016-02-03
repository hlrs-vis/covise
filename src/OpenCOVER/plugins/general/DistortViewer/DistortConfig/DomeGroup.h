/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DOMEGROUP_H
#define DOMEGROUP_H

#include <QGroupBox>
#include "ui_DomeGroup.h"
#include "ScreenDome.h"
#include "Scene.h"

class MainWindow;

class DomeGroup : public QGroupBox
{
    Q_OBJECT

public:
    DomeGroup(MainWindow *parent = 0);
    ~DomeGroup();
    void init();
    void update();
    void setupLineEdits();
    void setupLineEditsArray();
    void setupConnects();

public slots:
    //allgemein
    void lineEditEditingFinished(void);
    //LineEdits
    void edtAzimAngleEditingFinished();
    void edtAzimResEditingFinished();
    void edtPolarAngleEditingFinished();
    void edtPolarResEditingFinished();
    void edtRadiusEditingFinished();
    //ChkState
    void chkGeoMeshClicked(bool);

private:
    Ui::DomeGroup ui;
    ScreenDome *screen;
    MainWindow *mainWindow;
    QVector<QLineEdit *> lineEditsVec;
};

#endif // DOMEGROUP_H
