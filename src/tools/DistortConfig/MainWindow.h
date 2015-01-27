/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MainWindow_H
#define MainWindow_H
#include <QMessageBox>
#include <QMainWindow>
#include "ui_MainWindow.h"

class Projector;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~MainWindow();
    Scene *getOsgScene()
    {
        return ui.osgViewer->getScene();
    };
    void setupUiElements();
    void setupLineEdits();
    void setupConnects();
    void makeLineEditArray();
    void setupGeoSettings(std::string shapeType);
    void setupErrMsgBox(QString detailMsg);
    void correctLineEdt(QLineEdit *lineEdt);
    void calcNewScreen();
    void update();
    void save();
    void load(std::string fileName);
    void newScene();
    void init();

public slots:
    //allgemein
    void lineEditEditingFinished(void);
    //MenuBar
    void menuNewClicked(void);
    void menuConfigClicked(void);
    void menuAboutClicked(void);
    void menuQuitClicked(void);
    void menuOpenClicked(void);
    void menuSaveAsClicked(void);
    void menuSaveClicked(void);
    void menuHelpClicked(void);
    //Projector Selection
    void btnNewProjClicked();
    void btnDelProjClicked();
    void projNumValueChanged(int);

    //ToolBox
    //- Geometry
    void comboBoxGeoChanged(QString);
    void edtGeoOriEditingFinished();
    void edtGeoPosEditingFinished();
    void edtGeoSizeEditingFinished();
    //- Projector
    void edtProjPosEditingFinished();
    void edtProjOriEditingFinished();
    void edtProjRotEditingFinished();
    void edtProjAREditingFinished();
    void edtProjPREditingFinished();
    void edtProjLsEditingFinished();
    void edtProjNearEditingFinished();
    void edtProjFarEditingFinished();
    //- Screen
    void radioBtnAutoCalc();
    void radioBtnManualSet();
    void edtScreenPlaneEditingFinished();
    void edtScreenPosEditingFinished();
    void edtScreenSizeEditingFinished();
    void btnScreenCalcClicked();
    //-Visualisation
    void chkVisBlendClicked(bool);
    void chkVisDistortClicked(bool);
    void toolBtnBlendImgClicked();

private:
    Projector *currentProj()
    {
        return ui.osgViewer->getScene()->getProjector(ui.spinBoxProjNum->value() - 1);
    };

    int selectedProjNum; //ausgewählte Projektor-Nummer

    Ui::MainWindowClass ui;
    QWidget *geoSettingsWidget;
    QVector<QLineEdit *> lineEditsDoubleVec;
};

#endif // MainWindow_H
