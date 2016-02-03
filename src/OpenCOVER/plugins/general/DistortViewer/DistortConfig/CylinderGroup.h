/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CYLINDERGROUP_H
#define CYLINDERGROUP_H

#include <QGroupBox>
#include "ui_CylinderGroup.h"
#include "ScreenCylinder.h"
#include "Scene.h"

class MainWindow;

class CylinderGroup : public QGroupBox
{
    Q_OBJECT

public:
    CylinderGroup(MainWindow *parent = 0);
    ~CylinderGroup();
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
    void edtHeightEditingFinished();
    void edtHeightResEditingFinished();
    void edtRadiusEditingFinished();
    //ChkState
    void chkGeoMeshClicked(bool);

private:
    Ui::CylinderGroup ui;
    ScreenCylinder *screen;
    MainWindow *mainWindow;
    QVector<QLineEdit *> lineEditsVec;
};

#endif // CYLINDERGROUP_H
