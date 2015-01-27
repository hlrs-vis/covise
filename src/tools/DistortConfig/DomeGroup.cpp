/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DomeGroup.h"
#include "MainWindow.h"
#include "HelpFuncs.h"

DomeGroup::DomeGroup(MainWindow *parent)
    : QGroupBox(parent)
    , mainWindow(parent)
{
    screen = dynamic_cast<ScreenDome *>(Scene::getScreen());
    ui.setupUi(this);
    setupLineEdits();
    setupLineEditsArray();
    setupConnects();
    init();
}

DomeGroup::~DomeGroup()
{
}

void DomeGroup::init()
{
    std::string var_str;
    HelpFuncs::FloatToString(screen->getAzimSegmentSize(), var_str);
    ui.lineEditAzimAngle->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getAzimResolution(), var_str);
    ui.lineEditAzimRes->setText(var_str.c_str());
    HelpFuncs::FloatToString(screen->getPolarSegmentSize(), var_str);
    ui.lineEditPolarAngle->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getPolarResolution(), var_str);
    ui.lineEditPolarRes->setText(var_str.c_str());
    HelpFuncs::FloatToString(screen->getRadius(), var_str);
    ui.lineEditRadius->setText(var_str.c_str());
    ui.chkGeoMesh->setChecked(screen->getStateMesh());
}

void DomeGroup::update()
{
    mainWindow->update();
}

void DomeGroup::setupLineEdits()
{
    QValidator *dValidator = new QDoubleValidator(this);
    ui.lineEditAzimAngle->setValidator(dValidator);
    ui.lineEditPolarAngle->setValidator(dValidator);
    ui.lineEditRadius->setValidator(dValidator);

    QValidator *iValidator = new QIntValidator(this);
    ui.lineEditPolarRes->setValidator(iValidator);
    ui.lineEditAzimRes->setValidator(iValidator);
}

void DomeGroup::setupLineEditsArray()
{
    lineEditsVec.push_back(ui.lineEditAzimAngle);
    lineEditsVec.push_back(ui.lineEditAzimRes);
    lineEditsVec.push_back(ui.lineEditPolarAngle);
    lineEditsVec.push_back(ui.lineEditPolarRes);
    lineEditsVec.push_back(ui.lineEditRadius);
}

void DomeGroup::setupConnects()
{
    for (unsigned int i = 0; i < lineEditsVec.size(); i++)
    {
        connect(lineEditsVec.at(i), SIGNAL(editingFinished()), this, SLOT(lineEditEditingFinished()));
    }

    connect(ui.lineEditAzimAngle, SIGNAL(editingFinished()), this, SLOT(edtAzimAngleEditingFinished()));
    connect(ui.lineEditAzimRes, SIGNAL(editingFinished()), this, SLOT(edtAzimResEditingFinished()));
    connect(ui.lineEditPolarAngle, SIGNAL(editingFinished()), this, SLOT(edtPolarAngleEditingFinished()));
    connect(ui.lineEditPolarRes, SIGNAL(editingFinished()), this, SLOT(edtPolarResEditingFinished()));
    connect(ui.lineEditRadius, SIGNAL(editingFinished()), this, SLOT(edtRadiusEditingFinished()));
    connect(ui.chkGeoMesh, SIGNAL(clicked(bool)), this, SLOT(chkGeoMeshClicked(bool)));
}

//-------------------
#pragma region Slot_Funktionen
//-------------------
void DomeGroup::lineEditEditingFinished()
{
    //eingegebenen Text in formatierte Zahl umwandeln
    for (unsigned int i = 0; i < lineEditsVec.size(); i++)
    {
        //Zahlen formatieren
        double num = lineEditsVec.at(i)->text().toDouble();
        QString output = QString::number(num);
        lineEditsVec.at(i)->setText(output);
    }
}

void DomeGroup::edtAzimAngleEditingFinished()
{
    screen->setAzimSegmentSize(ui.lineEditAzimAngle->text().toFloat());
    update();
}

void DomeGroup::edtAzimResEditingFinished()
{
    screen->setAzimResolution(ui.lineEditAzimRes->text().toInt());
    update();
}

void DomeGroup::edtPolarAngleEditingFinished()
{
    screen->setPolarSegmentSize(ui.lineEditPolarAngle->text().toFloat());
    update();
}

void DomeGroup::edtPolarResEditingFinished()
{
    screen->setPolarResolution(ui.lineEditPolarRes->text().toInt());
    update();
}

void DomeGroup::edtRadiusEditingFinished()
{
    screen->setRadius(ui.lineEditRadius->text().toFloat());
    update();
}

void DomeGroup::chkGeoMeshClicked(bool state)
{
    screen->setStateMesh(state);
    update();
}

#pragma endregion