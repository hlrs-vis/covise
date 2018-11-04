/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SettingsDialog.h"
#include "Settings.h"
#include "HelpFuncs.h"
#include <QFileDialog>
#include <QIntValidator>

SettingsDialog::SettingsDialog(QWidget *parent)
    : QDialog(parent)
{
    setupGui();
    update();
}

SettingsDialog::~SettingsDialog()
{
}

void SettingsDialog::update()
{
    std::string var_Str;

    ui.lineEditBlendingPath->setText(Settings::getInstance()->imagePath.c_str());
    ui.lineEditFragFile->setText(Settings::getInstance()->fragShaderFile.c_str());
    ui.lineEditVertexFile->setText(Settings::getInstance()->vertShaderFile.c_str());

    HelpFuncs::IntToString(Settings::getInstance()->visResolutionH, var_Str);
    ui.lineEditResH->setText(var_Str.c_str());

    HelpFuncs::IntToString(Settings::getInstance()->visResolutionW, var_Str);
    ui.lineEditResW->setText(var_Str.c_str());
}

void SettingsDialog::setupGui()
{
    ui.setupUi(this);
    setupConnects();
    setupLineEdits();
}

void SettingsDialog::setupLineEdits()
{
    QValidator *pValidator = new QIntValidator(this);
    ui.lineEditResH->setValidator(pValidator);
    ui.lineEditResW->setValidator(pValidator);
}

void SettingsDialog::setupConnects()
{
    connect(ui.toolBtnBlendingPath, SIGNAL(clicked()), this, SLOT(btnBlendingPathClicked()));
    connect(ui.toolBtnFragFile, SIGNAL(clicked()), this, SLOT(btnFragFileClicked()));
    connect(ui.toolBtnVertexFile, SIGNAL(clicked()), this, SLOT(btnVertexFileClicked()));
    connect(ui.pushBtnOK, SIGNAL(clicked()), this, SLOT(btnOkClicked()));
    connect(ui.pushBtnCancel, SIGNAL(clicked()), this, SLOT(btnCancelClicked()));
}

void SettingsDialog::btnBlendingPathClicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    "./",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    //Benutzer abbruch
    if (dir.isEmpty())
        return;

    ui.lineEditBlendingPath->setText(dir);
}

void SettingsDialog::btnFragFileClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Fragment-Shader File"),
                                                    "./",
                                                    tr("Fragment-Shader File (*.frag)"));
    //Benutzer abbruch
    if (fileName.isEmpty())
        return;

    ui.lineEditFragFile->setText(fileName);
}

void SettingsDialog::btnVertexFileClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Vertex-Shader File"),
                                                    "./",
                                                    tr("Vertex-Shader File (*.vert)"));
    //Benutzer abbruch
    if (fileName.isEmpty())
        return;

    ui.lineEditVertexFile->setText(fileName);
}

void SettingsDialog::btnOkClicked()
{
    Settings::getInstance()->imagePath = ui.lineEditBlendingPath->text().toStdString();
    Settings::getInstance()->fragShaderFile = ui.lineEditFragFile->text().toStdString();
    Settings::getInstance()->vertShaderFile = ui.lineEditVertexFile->text().toStdString();
    Settings::getInstance()->visResolutionH = ui.lineEditResH->text().toInt();
    Settings::getInstance()->visResolutionW = ui.lineEditResW->text().toInt();
    close();
}

void SettingsDialog::btnCancelClicked()
{
    close();
}
