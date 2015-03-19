/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CylinderGroup.h"
#include "MainWindow.h"
#include "HelpFuncs.h"

CylinderGroup::CylinderGroup(MainWindow *parent)
    : QGroupBox(parent)
    , mainWindow(parent)
{
    screen = dynamic_cast<ScreenCylinder *>(Scene::getScreen());
    ui.setupUi(this);
    setupLineEdits();
    setupLineEditsArray();
    setupConnects();
    init();
}

CylinderGroup::~CylinderGroup()
{
}

void CylinderGroup::init()
{
    std::string var_str;
    HelpFuncs::FloatToString(screen->getHeight(), var_str);
    ui.lineEditHeight->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getZResolution(), var_str);
    ui.lineEditHeightRes->setText(var_str.c_str());
    HelpFuncs::FloatToString(screen->getSegmentSize(), var_str);
    ui.lineEditAzimAngle->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getAzimResolution(), var_str);
    ui.lineEditAzimRes->setText(var_str.c_str());
    HelpFuncs::FloatToString(screen->getRadius(), var_str);
    ui.lineEditRadius->setText(var_str.c_str());
    ui.chkGeoMesh->setChecked(screen->getStateMesh());
}
void CylinderGroup::update()
{
    mainWindow->update();
}

void CylinderGroup::setupLineEdits()
{
    QValidator *dValidator = new QDoubleValidator(this);
    ui.lineEditAzimAngle->setValidator(dValidator);
    ui.lineEditHeight->setValidator(dValidator);
    ui.lineEditRadius->setValidator(dValidator);

    QValidator *iValidator = new QIntValidator(this);
    ui.lineEditHeightRes->setValidator(iValidator);
    ui.lineEditAzimRes->setValidator(iValidator);
}

void CylinderGroup::setupLineEditsArray()
{
    lineEditsVec.push_back(ui.lineEditAzimAngle);
    lineEditsVec.push_back(ui.lineEditAzimRes);
    lineEditsVec.push_back(ui.lineEditHeight);
    lineEditsVec.push_back(ui.lineEditHeightRes);
    lineEditsVec.push_back(ui.lineEditRadius);
}

void CylinderGroup::setupConnects()
{
    for (int i = 0; i < lineEditsVec.size(); i++)
    {
        connect(lineEditsVec.at(i), SIGNAL(editingFinished()), this, SLOT(lineEditEditingFinished()));
    }

    connect(ui.lineEditAzimAngle, SIGNAL(editingFinished()), this, SLOT(edtAzimAngleEditingFinished()));
    connect(ui.lineEditAzimRes, SIGNAL(editingFinished()), this, SLOT(edtAzimResEditingFinished()));
    connect(ui.lineEditHeight, SIGNAL(editingFinished()), this, SLOT(edtHeightEditingFinished()));
    connect(ui.lineEditHeightRes, SIGNAL(editingFinished()), this, SLOT(edtHeightResEditingFinished()));
    connect(ui.lineEditRadius, SIGNAL(editingFinished()), this, SLOT(edtRadiusEditingFinished()));
    connect(ui.chkGeoMesh, SIGNAL(clicked(bool)), this, SLOT(chkGeoMeshClicked(bool)));
}

//-------------------
#pragma region Slot_Funktionen
//-------------------
void CylinderGroup::lineEditEditingFinished()
{
    //eingegebenen Text in formatierte Zahl umwandeln
    for (int i = 0; i < lineEditsVec.size(); i++)
    {
        //Zahlen formatieren
        double num = lineEditsVec.at(i)->text().toDouble();
        QString output = QString::number(num);
        lineEditsVec.at(i)->setText(output);
    }
}

void CylinderGroup::edtAzimAngleEditingFinished()
{
    screen->setSegmentSize(ui.lineEditAzimAngle->text().toFloat());
    update();
}

void CylinderGroup::edtAzimResEditingFinished()
{
    screen->setAzimResolution(ui.lineEditAzimRes->text().toInt());
    update();
}

void CylinderGroup::edtHeightEditingFinished()
{
    screen->setHeight(ui.lineEditHeight->text().toFloat());
    update();
}

void CylinderGroup::edtHeightResEditingFinished()
{
    screen->setZResolution(ui.lineEditHeightRes->text().toInt());
    update();
}

void CylinderGroup::edtRadiusEditingFinished()
{
    screen->setRadius(ui.lineEditRadius->text().toFloat());
    update();
}

void CylinderGroup::chkGeoMeshClicked(bool state)
{
    screen->setStateMesh(state);
    update();
}

#pragma endregion
