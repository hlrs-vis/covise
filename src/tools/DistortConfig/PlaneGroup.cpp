/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PlaneGroup.h"
#include "MainWindow.h"
#include "HelpFuncs.h"

PlaneGroup::PlaneGroup(MainWindow *parent)
    : QGroupBox(parent)
    , mainWindow(parent)
{
    screen = dynamic_cast<ScreenPlane *>(Scene::getScreen());
    ui.setupUi(this);
    setupLineEdits();
    setupLineEditsArray();
    setupConnects();
    init();
}

PlaneGroup::~PlaneGroup()
{
}

void PlaneGroup::init()
{
    std::string var_str;
    HelpFuncs::FloatToString(screen->getHeight(), var_str);
    ui.lineEditHeight->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getHeightResolution(), var_str);
    ui.lineEditHeightRes->setText(var_str.c_str());
    HelpFuncs::FloatToString(screen->getWidth(), var_str);
    ui.lineEditWidth->setText(var_str.c_str());
    HelpFuncs::IntToString(screen->getWidthResolution(), var_str);
    ui.lineEditWidthRes->setText(var_str.c_str());
    ui.chkGeoMesh->setChecked(screen->getStateMesh());
}
void PlaneGroup::update()
{
    mainWindow->update();
}

void PlaneGroup::setupLineEdits()
{
    QValidator *dValidator = new QDoubleValidator(this);
    ui.lineEditHeight->setValidator(dValidator);
    ui.lineEditHeight->setValidator(dValidator);

    QValidator *iValidator = new QIntValidator(this);
    ui.lineEditHeightRes->setValidator(iValidator);
    ui.lineEditHeightRes->setValidator(iValidator);
}

void PlaneGroup::setupLineEditsArray()
{
    lineEditsVec.push_back(ui.lineEditHeight);
    lineEditsVec.push_back(ui.lineEditHeightRes);
    lineEditsVec.push_back(ui.lineEditHeight);
    lineEditsVec.push_back(ui.lineEditHeightRes);
}

void PlaneGroup::setupConnects()
{
    for (unsigned int i = 0; i < lineEditsVec.size(); i++)
    {
        connect(lineEditsVec.at(i), SIGNAL(editingFinished()), this, SLOT(lineEditEditingFinished()));
    }

    connect(ui.lineEditHeight, SIGNAL(editingFinished()), this, SLOT(edtHeightEditingFinished()));
    connect(ui.lineEditHeightRes, SIGNAL(editingFinished()), this, SLOT(edtHeightResEditingFinished()));
    connect(ui.lineEditWidth, SIGNAL(editingFinished()), this, SLOT(edtWidthEditingFinished()));
    connect(ui.lineEditWidthRes, SIGNAL(editingFinished()), this, SLOT(edtWidthResEditingFinished()));
    connect(ui.chkGeoMesh, SIGNAL(clicked(bool)), this, SLOT(chkGeoMeshClicked(bool)));
}

//-------------------
#pragma region Slot_Funktionen
//-------------------
void PlaneGroup::lineEditEditingFinished()
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

void PlaneGroup::edtWidthEditingFinished()
{
    screen->setWidth(ui.lineEditWidth->text().toFloat());
    update();
}

void PlaneGroup::edtWidthResEditingFinished()
{
    screen->setWidthResolution(ui.lineEditWidthRes->text().toInt());
    update();
}

void PlaneGroup::edtHeightEditingFinished()
{
    QString text = ui.lineEditHeight->text();
    float height = text.toFloat();
    float height2 = ui.lineEditHeight->text().toFloat();
    screen->setHeight(ui.lineEditHeight->text().toFloat());
    update();
}

void PlaneGroup::edtHeightResEditingFinished()
{
    screen->setHeightResolution(ui.lineEditHeightRes->text().toInt());
    update();
}

void PlaneGroup::chkGeoMeshClicked(bool state)
{
    screen->setStateMesh(state);
    update();
}

#pragma endregion
