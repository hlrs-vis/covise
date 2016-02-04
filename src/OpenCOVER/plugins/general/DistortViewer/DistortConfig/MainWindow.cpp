/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//Covise -> winsock2.h in Covise-Includes MUSS ausnahmsweise vor *Windows.h includiert werden
#include "XmlTools.h"
#include "HelpFuncs.h"
#include "Settings.h"
#include <QFileDialog>
//Local
#include "MainWindow.h"
#include "PlaneGroup.h"
#include "CylinderGroup.h"
#include "DomeGroup.h"
#include "OpenGeoGroup.h"
#include "AboutDialog.h"
#include "SettingsDialog.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setupUiElements();
    init();
}

MainWindow::~MainWindow()
{
}

void MainWindow::init()
{
    //ersten Projektor aktivieren
    ui.osgViewer->getScene()->getProjector(0)->active = true;

    //GeoSettings
    geoSettingsWidget = new DomeGroup(this);
    std::string shapeType = ui.osgViewer->getScene()->getScreen()->getShapeType();
    int shapeIndex = ui.comboBoxGeo->findText(shapeType.c_str());
    ui.comboBoxGeo->setCurrentIndex(shapeIndex);
    setupGeoSettings(shapeType);

    //Daten in Edit-Felder eintragen
    update();
}

void MainWindow::update()
{
    std::string var_str;

    //Da Screengeo und Projektorpos evt. verändert -> update Visgroup
    if (Scene::getVisStatus())
        currentProj()->getVisScene()->updateVisGroup();
    //Wenn autoCalc=true wird optimale Screenpos. berechnet
    calcNewScreen();

    //Projector-Control
    int projCount = ui.osgViewer->getScene()->getNumProjectors();
    ui.spinBoxProjNum->setRange(1, projCount);
    ui.horSliderProjNum->setRange(1, projCount);
    if (projCount <= 1)
        ui.btnDelProj->setDisabled(true);
    else
        ui.btnDelProj->setDisabled(false);

    //Toolbox updaten
    //- Geometry
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getOrientation().x(), var_str);
    ui.edtGeoOriX->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getOrientation().y(), var_str);
    ui.edtGeoOriY->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getOrientation().z(), var_str);
    ui.edtGeoOriZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getCenterPos().x(), var_str);
    ui.edtGeoPosX->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getCenterPos().y(), var_str);
    ui.edtGeoPosY->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getCenterPos().z(), var_str);
    ui.edtGeoPosZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getScaleVec().x(), var_str);
    ui.edtGeoSizeX->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getScaleVec().y(), var_str);
    ui.edtGeoSizeY->setText(var_str.c_str());
    HelpFuncs::FloatToString(ui.osgViewer->getScene()->getScreen()->getScaleVec().z(), var_str);
    ui.edtGeoSizeZ->setText(var_str.c_str());

    //- Projector
    selectedProjNum = ui.spinBoxProjNum->value() - 1;
    HelpFuncs::FloatToString(currentProj()->getPosition().x(), var_str);
    ui.edtProjPosX->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getPosition().y(), var_str);
    ui.edtProjPosY->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getPosition().z(), var_str);
    ui.edtProjPosZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getProjDirection().x(), var_str);
    ui.edtProjOriX->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getProjDirection().y(), var_str);
    ui.edtProjOriY->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getProjDirection().z(), var_str);
    ui.edtProjOriZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getUpDirection().x(), var_str);
    ui.edtProjRotX->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getUpDirection().y(), var_str);
    ui.edtProjRotY->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getUpDirection().z(), var_str);
    ui.edtProjRotZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getAspectRatioH(), var_str);
    ui.edtProjARh->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getAspectRatioW(), var_str);
    ui.edtProjARw->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getProjRatio(), var_str);
    ui.edtProjPR->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getLensShiftH() * 100, var_str);
    ui.edtProjLsH->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getLensShiftV() * 100, var_str);
    ui.edtProjLsV->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getFarClipping(), var_str);
    ui.edtProjFar->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getNearClipping(), var_str);
    ui.edtProjNear->setText(var_str.c_str());

    //- Screen
    bool autoCalc = currentProj()->getAutoCalc();
    ui.radioBtnAuto->setChecked(autoCalc);
    ui.radioBtnManual->setChecked(!autoCalc);
    ui.edtScreenPlaneA_4->setEnabled(!autoCalc);
    ui.edtScreenPlaneB_4->setEnabled(!autoCalc);
    ui.edtScreenPlaneC_4->setEnabled(!autoCalc);
    ui.edtScreenPlaneD_4->setEnabled(!autoCalc);
    ui.btnScreenCalc->setEnabled(!autoCalc);

    ui.edtScreenPlaneA_4->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenPlane().asVec4().x(), var_str);
    ui.edtScreenPlaneA_4->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenPlane().asVec4().y(), var_str);
    ui.edtScreenPlaneB_4->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenPlane().asVec4().z(), var_str);
    ui.edtScreenPlaneC_4->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenPlane().asVec4().w(), var_str);
    ui.edtScreenPlaneD_4->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getEulerAngles().x(), var_str);
    ui.edtScreenEulerH->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getEulerAngles().y(), var_str);
    ui.edtScreenEulerP->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getEulerAngles().z(), var_str);
    ui.edtScreenEulerR->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getScreenCenter().x(), var_str);
    ui.edtScreenPosX->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenCenter().y(), var_str);
    ui.edtScreenPosY->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenCenter().z(), var_str);
    ui.edtScreenPosZ->setText(var_str.c_str());

    HelpFuncs::FloatToString(currentProj()->getScreenHeight(), var_str);
    ui.edtScreenH->setText(var_str.c_str());
    HelpFuncs::FloatToString(currentProj()->getScreenWidth(), var_str);
    ui.edtScreenW->setText(var_str.c_str());

    //- Visualisation
    ui.chkVisBlend->setChecked(currentProj()->getVisScene()->getBlendState());
    ui.chkVisDistort->setChecked(currentProj()->getVisScene()->getDistortState());
    ui.lineEditBlendImg->setText(currentProj()->getVisScene()->getBlendImgFilePath().c_str());

    //OSG-Viewer
    ui.osgViewer->getScene()->updateScene();
}

void MainWindow::newScene()
{
    ui.osgViewer->getScene()->resetSceneContent();
    ui.osgViewer->getScene()->init();
    ui.spinBoxProjNum->setValue(1);

    //Update wird durch valueChange() event ausgeführt
}

void MainWindow::save()
{
    Settings::getInstance()->saveToXML();
    ui.osgViewer->getScene()->saveToXML();
}

void MainWindow::load(std::string fileName)
{
    XmlTools::getInstance()->setNewConfigFile(fileName);
    newScene();
    Settings::getInstance()->loadFromXML();
    ui.osgViewer->getScene()->loadFromXML();
}

void MainWindow::correctLineEdt(QLineEdit *lineEdt)
{
    lineEdt->undo();
}

void MainWindow::calcNewScreen()
{
    if (ui.radioBtnAuto->isChecked())
    {
        osg::Plane newScreenPlane = currentProj()->calcScreenPlane();
        currentProj()->setScreenPlane(newScreenPlane);
        currentProj()->calcScreen();
    }
}
void MainWindow::setupErrMsgBox(QString infoMsg)
{
    //Messagebox erstellen
    QMessageBox valErrMsgBox;
    valErrMsgBox.setText("Invalid value.");
    valErrMsgBox.setIcon(QMessageBox::Information);
    valErrMsgBox.setInformativeText(infoMsg);
    valErrMsgBox.exec();
}
void MainWindow::setupUiElements()
{
    //ComboBox
    QStringList comboList;
    comboList << "Dome"
              << "Cylinder"
              << "Plane"
              << "Custom Geometry";
    ui.comboBoxGeo->addItems(comboList);

    setupLineEdits();
    setupConnects();
}
void MainWindow::makeLineEditArray()
{
    //Toolbox
    //- Geometry
    lineEditsDoubleVec.push_back(ui.edtGeoOriX);
    lineEditsDoubleVec.push_back(ui.edtGeoOriY);
    lineEditsDoubleVec.push_back(ui.edtGeoOriZ);
    lineEditsDoubleVec.push_back(ui.edtGeoPosX);
    lineEditsDoubleVec.push_back(ui.edtGeoPosY);
    lineEditsDoubleVec.push_back(ui.edtGeoPosZ);
    lineEditsDoubleVec.push_back(ui.edtGeoSizeX);
    lineEditsDoubleVec.push_back(ui.edtGeoSizeY);
    lineEditsDoubleVec.push_back(ui.edtGeoSizeZ);
    // -Projector
    lineEditsDoubleVec.push_back(ui.edtProjPosX);
    lineEditsDoubleVec.push_back(ui.edtProjPosY);
    lineEditsDoubleVec.push_back(ui.edtProjPosZ);
    lineEditsDoubleVec.push_back(ui.edtProjOriX);
    lineEditsDoubleVec.push_back(ui.edtProjOriY);
    lineEditsDoubleVec.push_back(ui.edtProjOriZ);
    lineEditsDoubleVec.push_back(ui.edtProjRotX);
    lineEditsDoubleVec.push_back(ui.edtProjRotY);
    lineEditsDoubleVec.push_back(ui.edtProjRotZ);
    lineEditsDoubleVec.push_back(ui.edtProjARw);
    lineEditsDoubleVec.push_back(ui.edtProjARh);
    lineEditsDoubleVec.push_back(ui.edtProjLsH);
    lineEditsDoubleVec.push_back(ui.edtProjLsV);
    lineEditsDoubleVec.push_back(ui.edtProjNear);
    lineEditsDoubleVec.push_back(ui.edtProjFar);
    lineEditsDoubleVec.push_back(ui.edtProjPR);
    //- Screen
    lineEditsDoubleVec.push_back(ui.edtScreenPlaneA_4);
    lineEditsDoubleVec.push_back(ui.edtScreenPlaneB_4);
    lineEditsDoubleVec.push_back(ui.edtScreenPlaneC_4);
    lineEditsDoubleVec.push_back(ui.edtScreenPlaneD_4);
    lineEditsDoubleVec.push_back(ui.edtScreenPosX);
    lineEditsDoubleVec.push_back(ui.edtScreenPosY);
    lineEditsDoubleVec.push_back(ui.edtScreenPosZ);
    lineEditsDoubleVec.push_back(ui.edtScreenH);
    lineEditsDoubleVec.push_back(ui.edtScreenW);
}

void MainWindow::setupLineEdits()
{
    makeLineEditArray();

    QValidator *pValidator = new QDoubleValidator(this);
    for (int i = 0; i < lineEditsDoubleVec.size(); i++)
    {
        lineEditsDoubleVec.at(i)->setValidator(pValidator);
    }
}

void MainWindow::setupConnects()
{
    for (int i = 0; i < lineEditsDoubleVec.size(); i++)
    {
        connect(lineEditsDoubleVec.at(i), SIGNAL(editingFinished()), this, SLOT(lineEditEditingFinished()));
    }

    //MenuBar
    connect(ui.actionNew, SIGNAL(triggered()), this, SLOT(menuNewClicked()));
    connect(ui.actionSettings, SIGNAL(triggered()), this, SLOT(menuConfigClicked()));
    connect(ui.actionAbout, SIGNAL(triggered()), this, SLOT(menuAboutClicked()));
    connect(ui.actionShow_Help, SIGNAL(triggered()), this, SLOT(menuHelpClicked()));
    connect(ui.actionQuit, SIGNAL(triggered()), this, SLOT(menuQuitClicked()));
    connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(menuOpenClicked()));
    connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(menuSaveClicked()));
    connect(ui.actionSave_as, SIGNAL(triggered()), this, SLOT(menuSaveAsClicked()));
    //ProjektorAuswahl
    connect(ui.btnNewProj, SIGNAL(clicked()), this, SLOT(btnNewProjClicked()));
    connect(ui.btnDelProj, SIGNAL(clicked()), this, SLOT(btnDelProjClicked()));
    connect(ui.spinBoxProjNum, SIGNAL(valueChanged(int)), this, SLOT(projNumValueChanged(int)));
    //ToolBox
    //- Geometry
    connect(ui.comboBoxGeo, SIGNAL(currentIndexChanged(QString)), this, SLOT(comboBoxGeoChanged(QString)));
    connect(ui.edtGeoOriX, SIGNAL(editingFinished()), this, SLOT(edtGeoOriEditingFinished()));
    connect(ui.edtGeoOriY, SIGNAL(editingFinished()), this, SLOT(edtGeoOriEditingFinished()));
    connect(ui.edtGeoOriZ, SIGNAL(editingFinished()), this, SLOT(edtGeoOriEditingFinished()));
    connect(ui.edtGeoPosX, SIGNAL(editingFinished()), this, SLOT(edtGeoPosEditingFinished()));
    connect(ui.edtGeoPosY, SIGNAL(editingFinished()), this, SLOT(edtGeoPosEditingFinished()));
    connect(ui.edtGeoPosZ, SIGNAL(editingFinished()), this, SLOT(edtGeoPosEditingFinished()));
    connect(ui.edtGeoSizeX, SIGNAL(editingFinished()), this, SLOT(edtGeoSizeEditingFinished()));
    connect(ui.edtGeoSizeY, SIGNAL(editingFinished()), this, SLOT(edtGeoSizeEditingFinished()));
    connect(ui.edtGeoSizeZ, SIGNAL(editingFinished()), this, SLOT(edtGeoSizeEditingFinished()));
    //- Projector
    connect(ui.edtProjPosX, SIGNAL(editingFinished()), this, SLOT(edtProjPosEditingFinished()));
    connect(ui.edtProjPosY, SIGNAL(editingFinished()), this, SLOT(edtProjPosEditingFinished()));
    connect(ui.edtProjPosZ, SIGNAL(editingFinished()), this, SLOT(edtProjPosEditingFinished()));
    connect(ui.edtProjOriX, SIGNAL(editingFinished()), this, SLOT(edtProjOriEditingFinished()));
    connect(ui.edtProjOriY, SIGNAL(editingFinished()), this, SLOT(edtProjOriEditingFinished()));
    connect(ui.edtProjOriZ, SIGNAL(editingFinished()), this, SLOT(edtProjOriEditingFinished()));
    connect(ui.edtProjRotX, SIGNAL(editingFinished()), this, SLOT(edtProjRotEditingFinished()));
    connect(ui.edtProjRotY, SIGNAL(editingFinished()), this, SLOT(edtProjRotEditingFinished()));
    connect(ui.edtProjRotZ, SIGNAL(editingFinished()), this, SLOT(edtProjRotEditingFinished()));
    connect(ui.edtProjARw, SIGNAL(editingFinished()), this, SLOT(edtProjAREditingFinished()));
    connect(ui.edtProjARh, SIGNAL(editingFinished()), this, SLOT(edtProjAREditingFinished()));
    connect(ui.edtProjPR, SIGNAL(editingFinished()), this, SLOT(edtProjPREditingFinished()));
    connect(ui.edtProjLsH, SIGNAL(editingFinished()), this, SLOT(edtProjLsEditingFinished()));
    connect(ui.edtProjLsV, SIGNAL(editingFinished()), this, SLOT(edtProjLsEditingFinished()));
    connect(ui.edtProjNear, SIGNAL(editingFinished()), this, SLOT(edtProjNearEditingFinished()));
    connect(ui.edtProjFar, SIGNAL(editingFinished()), this, SLOT(edtProjFarEditingFinished()));
    //- Screen
    connect(ui.radioBtnAuto, SIGNAL(clicked()), this, SLOT(radioBtnAutoCalc()));
    connect(ui.radioBtnManual, SIGNAL(clicked()), this, SLOT(radioBtnManualSet()));
    connect(ui.edtScreenPlaneA_4, SIGNAL(editingFinished()), this, SLOT(edtScreenPlaneEditingFinished()));
    connect(ui.edtScreenPlaneB_4, SIGNAL(editingFinished()), this, SLOT(edtScreenPlaneEditingFinished()));
    connect(ui.edtScreenPlaneC_4, SIGNAL(editingFinished()), this, SLOT(edtScreenPlaneEditingFinished()));
    connect(ui.edtScreenPlaneD_4, SIGNAL(editingFinished()), this, SLOT(edtScreenPlaneEditingFinished()));
    connect(ui.edtScreenPosX, SIGNAL(editingFinished()), this, SLOT(edtScreenPosEditingFinished()));
    connect(ui.edtScreenPosY, SIGNAL(editingFinished()), this, SLOT(edtScreenPosEditingFinished()));
    connect(ui.edtScreenPosZ, SIGNAL(editingFinished()), this, SLOT(edtScreenPosEditingFinished()));
    connect(ui.edtScreenH, SIGNAL(editingFinished()), this, SLOT(edtScreenSizeEditingFinished()));
    connect(ui.edtScreenW, SIGNAL(editingFinished()), this, SLOT(edtScreenSizeEditingFinished()));
    connect(ui.btnScreenCalc, SIGNAL(clicked()), this, SLOT(btnScreenCalcClicked()));
    //- Visualisation
    connect(ui.chkVisBlend, SIGNAL(clicked(bool)), this, SLOT(chkVisBlendClicked(bool)));
    connect(ui.chkVisDistort, SIGNAL(clicked(bool)), this, SLOT(chkVisDistortClicked(bool)));
    connect(ui.toolBtnBlendImg, SIGNAL(clicked()), this, SLOT(toolBtnBlendImgClicked()));
}

void MainWindow::setupGeoSettings(std::string shapeType)
{
    delete geoSettingsWidget;
    if (!shapeType.empty())
    {
        if (strcasecmp(shapeType.c_str(), "Plane") == 0)
            geoSettingsWidget = new PlaneGroup(this);
        else
        {
            if (strcasecmp(shapeType.c_str(), "Cylinder") == 0)
                geoSettingsWidget = new CylinderGroup(this);
            else
            {
                if (strcasecmp(shapeType.c_str(), "Dome") == 0)
                    geoSettingsWidget = new DomeGroup(this);
                else
                {
                    if (strcasecmp(shapeType.c_str(), "Custom Geometry") == 0)
                        geoSettingsWidget = new OpenGeoGroup(this);
                }
            }
        }
        ui.verticalLayout_PlaceHolder->addWidget(geoSettingsWidget);
    }
}
//-------------------
#pragma region Slot_Funktionen
//-------------------

void MainWindow::lineEditEditingFinished()
{
    //eingegebenen Text in formatierte Zahl umwandeln
    for (int i = 0; i < lineEditsDoubleVec.size(); i++)
    {
        double num = lineEditsDoubleVec.at(i)->text().toDouble();
        QString output = QString::number(num);
        lineEditsDoubleVec.at(i)->setText(output);
    }
}

void MainWindow::menuNewClicked()
{
    newScene();
}

void MainWindow::menuQuitClicked(void)
{
    close();
}

void MainWindow::menuConfigClicked(void)
{
    SettingsDialog settingsDialog;
    settingsDialog.exec();
}

void MainWindow::menuAboutClicked()
{
    AboutDialog aboutDialog;
    aboutDialog.exec();
}

void MainWindow::menuOpenClicked(void)
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "./",
                                                    tr("ConfigFile (*.xml)"));
    //Benutzer abbruch
    if (fileName.isEmpty())
        return;

    load(fileName.toStdString());
    update();
}

void MainWindow::menuSaveAsClicked(void)
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
                                                    "./config.xml",
                                                    tr("ConfigFile (*.xml)"));

    //Benutzer abbruch
    if (fileName.isEmpty())
        return;

    XmlTools::getInstance()->setNewConfigFile(fileName.toStdString());
    save();
}

void MainWindow::menuSaveClicked(void)
{
    save();
    update();
}

void MainWindow::menuHelpClicked(void)
{
}

void MainWindow::btnNewProjClicked()
{
    //neuen Projektor hinzufügen
    ui.osgViewer->getScene()->makeNewProjector();

    //UI-Projektorelemente updaten
    int projCount = ui.osgViewer->getScene()->getNumProjectors();
    ui.spinBoxProjNum->setRange(1, projCount);
    ui.horSliderProjNum->setRange(1, projCount);
    ui.spinBoxProjNum->setValue(projCount);

    //Durch SetValue wird Slot ValueChange() aufgerufen -> load()
}

void MainWindow::btnDelProjClicked()
{
    //ausgewählte Projektornummer (eins kleiner wie Spinbox Value)
    int delNum = ui.spinBoxProjNum->value() - 1;

    //ausgewählten Projektor löschen
    ui.osgViewer->getScene()->deleteProjector(delNum);

    //UI-Projektorelemente updaten
    ui.spinBoxProjNum->setValue(ui.spinBoxProjNum->value() - 1);

    //Durch setValue neuerWert -> Funktion ValueChange wird ausgeführt
}

void MainWindow::projNumValueChanged(int newValue)
{
    //Vorherigen Projektor deaktivieren, sofern noch existent
    //selectedProjNum: hier Projektornummer vor ValueChange
    if (selectedProjNum <= (ui.osgViewer->getScene()->getNumProjectors() - 1))
        currentProj()->active = false;
    //Ausgewählten Projektor aktivieren
    ui.osgViewer->getScene()->getProjector(newValue - 1)->active = true; //newValue-1, um auf Projektor-Nummer zu kommen

    //UI-Felder akzalisieren
    update();
}

void MainWindow::comboBoxGeoChanged(QString shapeType)
{
    //Screen-Typ erstellen
    bool valid = ui.osgViewer->getScene()->setScreenShape(shapeType.toStdString());

    //Gui an Screentyp anpassen
    setupGeoSettings(shapeType.toStdString());

    if (shapeType == "Custom Geometry")
    {
        OpenGeoGroup *settingsWidgetCustom = dynamic_cast<OpenGeoGroup *>(geoSettingsWidget);
        if (!dynamic_cast<ScreenLoad *>(ui.osgViewer->getScene()->getScreen())->fileIsValid())
        {
            if (!settingsWidgetCustom->toolButtonClicked())
                valid = false;
        }
    }

    if (!valid)
    {
        //Auf standard Geometrie "Dome" zurücksetzen
        ui.comboBoxGeo->setCurrentIndex(ui.comboBoxGeo->findText("Dome"));
        comboBoxGeoChanged("Dome");
        return;
    }
    //Alles ok!
    update();
}

void MainWindow::edtGeoOriEditingFinished()
{
    osg::Vec3 geoOri = osg::Vec3(ui.edtGeoOriX->text().toFloat(),
                                 ui.edtGeoOriY->text().toFloat(),
                                 ui.edtGeoOriZ->text().toFloat());
    ui.osgViewer->getScene()->getScreen()->setOrientation(geoOri);
    update();
}

void MainWindow::edtGeoPosEditingFinished()
{
    osg::Vec3 geoPos = osg::Vec3(ui.edtGeoPosX->text().toFloat(),
                                 ui.edtGeoPosY->text().toFloat(),
                                 ui.edtGeoPosZ->text().toFloat());
    ui.osgViewer->getScene()->getScreen()->setCenterPos(geoPos);
    update();
}

void MainWindow::edtGeoSizeEditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());
    float chkValue = lineEdit->text().toFloat();

    //- Scaling der Projektionsgeometrie > 0
    if (chkValue <= 0)
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("Geometry scale value must be bigger than 0.0!");
    }
    else
    {
        osg::Vec3 geoSize = osg::Vec3(ui.edtGeoSizeX->text().toFloat(),
                                      ui.edtGeoSizeY->text().toFloat(),
                                      ui.edtGeoSizeZ->text().toFloat());

        ui.osgViewer->getScene()->getScreen()->setScaleVec(geoSize);
        update();
    }
}

void MainWindow::edtProjPosEditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());

    osg::Vec3 projPos = osg::Vec3(ui.edtProjPosX->text().toFloat(),
                                  ui.edtProjPosY->text().toFloat(),
                                  ui.edtProjPosZ->text().toFloat());

    //- ViewPoint des Projektors muss unterschiedlich zu Position sein
    if (projPos == currentProj()->getProjDirection())
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("Projector position must be different to view-point vector!");
    }
    else
    {
        currentProj()->setPosition(projPos);
        update();
    }
}

void MainWindow::edtProjOriEditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());

    osg::Vec3 projOri = osg::Vec3(ui.edtProjOriX->text().toFloat(),
                                  ui.edtProjOriY->text().toFloat(),
                                  ui.edtProjOriZ->text().toFloat());

    //- ViewPoint des Projektors muss unterschiedlich zu Position sein
    if (projOri == currentProj()->getPosition())
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("Projector position must be different to view-point vector!");
    }
    else
    {
        currentProj()->setProjDirection(projOri);
        update();
    }
}

void MainWindow::edtProjRotEditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());

    osg::Vec3 projRot = osg::Vec3(ui.edtProjRotX->text().toFloat(),
                                  ui.edtProjRotY->text().toFloat(),
                                  ui.edtProjRotZ->text().toFloat());

    //Mind. eine Komponente des up-direction Vektors des Projektors darf nicht 0 sein
    unsigned int i, val_count;
    val_count = 0;
    for (i = 0; i < projRot.num_components; i++)
    {
        if (projRot[i] == 0.0f)
            val_count++;
    }
    if (val_count == projRot.num_components)
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("At least one value of the Projector's up-direction vector must be bigger than 0.0!");
    }
    else
    {
        currentProj()->setUpDirection(projRot);
        update();
    }
}

void MainWindow::edtProjAREditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());
    float chkValue = lineEdit->text().toFloat();

    //Seitenverhältnis des Projektors > 0
    if (chkValue <= 0)
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("The aspect ratio must be bigger than 0.0!");
        ;
    }
    else
    {
        currentProj()->setAspectRatioH(ui.edtProjARh->text().toFloat());
        currentProj()->setAspectRatioW(ui.edtProjARw->text().toFloat());
        update();
    }
}

void MainWindow::edtProjPREditingFinished()
{
    float projPR = ui.edtProjPR->text().toFloat();

    //Projektionsverhältnis des Projektors > 0
    if (projPR <= 0)
    {
        correctLineEdt(ui.edtProjPR);
        setupErrMsgBox("The aspect ratio of the Projector must be bigger than 0.0!");
    }
    else
    {
        currentProj()->setProjRatio(projPR);
        update();
    }
}

void MainWindow::edtProjLsEditingFinished()
{
    float projLSh = ui.edtProjLsH->text().toFloat();
    float projLSv = ui.edtProjLsV->text().toFloat();
    currentProj()->setLensShiftH(projLSh / 100);
    currentProj()->setLensShiftV(projLSv / 100);
    update();
}

void MainWindow::edtProjNearEditingFinished()
{
    float projFar = ui.edtProjFar->text().toFloat();
    float projNear = ui.edtProjNear->text().toFloat();

    //Clippingebenen des Frustums müssen größer null sein
    if (projNear <= 0)
    {
        correctLineEdt(ui.edtProjNear);
        setupErrMsgBox("The frustum clipping-plane must be bigger than 0.0!");
        return;
    }

    //Far-Clippingebene des Frustums muss größer sein als Near-Clippingebene
    if (projNear >= projFar)
    {
        correctLineEdt(ui.edtProjNear);
        setupErrMsgBox("The far clipping plane of the Projector must be bigger than the near one!");
        return;
    }

    currentProj()->setNearClipping(projNear);
    update();
}

void MainWindow::edtProjFarEditingFinished()
{
    float projFar = ui.edtProjFar->text().toFloat();
    float projNear = ui.edtProjNear->text().toFloat();

    //Clippingebenen des Frustums müssen größer null sein
    if (projFar <= 0)
    {
        correctLineEdt(ui.edtProjFar);
        setupErrMsgBox("The frustum clipping-plane must be bigger than 0.0!");
        return;
    }

    //Far-Clippingebene des Frustums muss größer sein als Near-Clippingebene
    if (projNear >= projFar)
    {
        correctLineEdt(ui.edtProjFar);
        setupErrMsgBox("The far clipping plane of the Projector must be bigger than the near one!");
        return;
    }

    currentProj()->setFarClipping(projFar);
    update();
}

void MainWindow::radioBtnAutoCalc()
{
    currentProj()->setAutoCalc(true);
    //UI-Elemente deaktivieren
    update();
}

void MainWindow::radioBtnManualSet()
{
    currentProj()->setAutoCalc(false);
    //UI-Elemente aktivieren
    update();
}

void MainWindow::edtScreenPlaneEditingFinished()
{
    osg::Vec4 screenPlane = osg::Vec4(ui.edtScreenPlaneA_4->text().toFloat(),
                                      ui.edtScreenPlaneB_4->text().toFloat(),
                                      ui.edtScreenPlaneC_4->text().toFloat(),
                                      ui.edtScreenPlaneD_4->text().toFloat());
    currentProj()->setScreenPlane(osg::Plane(screenPlane));
    update();
}
void MainWindow::edtScreenPosEditingFinished()
{
    osg::Vec3 screenPos = osg::Vec3(ui.edtScreenPosX->text().toFloat(),
                                    ui.edtScreenPosY->text().toFloat(),
                                    ui.edtScreenPosZ->text().toFloat());
    currentProj()->setScreenCenter(screenPos);
    update();
}

void MainWindow::edtScreenSizeEditingFinished()
{
    QLineEdit *lineEdit = qobject_cast<QLineEdit *>(QObject::sender());
    float chkValue = lineEdit->text().toFloat();

    //Größe der virt. Projektionsfläche > 0
    if (chkValue <= 0)
    {
        correctLineEdt(lineEdit);
        setupErrMsgBox("The size of the virtual projection-screen must be bigger than 0.0!");
    }
    else
    {
        currentProj()->setScreenHeight(ui.edtScreenH->text().toFloat());
        currentProj()->setScreenWidth(ui.edtScreenW->text().toFloat());
        update();
    }
}

void MainWindow::btnScreenCalcClicked()
{
    currentProj()->calcScreen();
    update();
}

void MainWindow::toolBtnBlendImgClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image-File"),
                                                    Settings::getInstance()->imagePath.c_str(),
                                                    tr("Image-File (*.png)"));
    //Benutzer abbruch
    if (fileName.isEmpty())
        return;

    currentProj()->getVisScene()->setBlendImgFilePath(fileName.toStdString());
    update();
}

void MainWindow::chkVisBlendClicked(bool state)
{
    currentProj()->getVisScene()->setBlendState(state);
}

void MainWindow::chkVisDistortClicked(bool state)
{
    currentProj()->getVisScene()->setDistortState(state);
}

#pragma endregion
