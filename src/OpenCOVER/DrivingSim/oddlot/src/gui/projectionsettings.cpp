/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/

#include "projectionsettings.hpp"
/*#include "projectwidget.hpp"
#include "src/mainwindow.hpp"*/
#include "ui_projectionsettings.h"
#include "src/data/projectdata.hpp"
#include "src/data/georeference.hpp"

// Data //

//ProjectionSettings *ProjectionSettings::inst = NULL;

void ProjectionSettings::okPressed()
{
    updateSettings();
}


void ProjectionSettings::PresetIndexChanged(const QString &change)
{
    for (int i = 0; i < presets.size(); i++)
    {
        if (change.toStdString() == presets[i].name)
        {
            ui->ProjectionEdit->setText(presets[i].to.c_str());
            ui->SourceEdit->setText(presets[i].from.c_str());
        }
    }
    updateUi();
}

//################//
// CONSTRUCTOR    //
//################//

ProjectionSettings::ProjectionSettings()
    : ui(new Ui::ProjectionSettings)
    , projectData_(nullptr)
{
    //inst = this;

    ui->setupUi(this);

    //connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
    connect(ui->presetBox, &QComboBox::currentTextChanged, [=](const QString &change){PresetIndexChanged(change);});

    //Initialize presetBox
    /*QStringList set;
    Preset pre = None;
    for(int i = None; i < WGS84_to_WGS84_Ellipsoid; i++)
    {
        pre += i;
        set += QVariant::fromValue(i).value<QString>();
    }*/
    //QString s = QVariant::fromValue(ProjectionSettings::None).value<QString>();

    ui->presetBox->addItem("None");

#ifdef WIN32
    char* pValue;
    size_t len;
    errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
    if (err || pValue == NULL || strlen(pValue) == 0)
        err = _dupenv_s(&pValue, &len, "COVISEDIR");
    if (err)
        pValue = "";
    std::string covisedir = pValue;
#else
    QString covisedir = getenv("ODDLOTDIR");
    if (covisedir == "")
        covisedir = getenv("COVISEDIR");
#endif
    std::string dir = covisedir + "/share/covise/"; 

    presets.push_back(PresetInfo("+proj=longlat +datum=WGS84", "+proj=utm +zone=32 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0", "UTM Zone 32"));
    presets.push_back(PresetInfo("+proj=longlat +datum=WGS84", "+proj=utm +zone=50 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0", "UTM Zone 50"));
    presets.push_back(PresetInfo("+proj=longlat +datum=WGS84", "+proj=utm +zone=45 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0", "UTM Zone 45"));
    presets.push_back(PresetInfo("+proj=longlat +datum=WGS84", "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam", "Gauss Krueger BW"));
    presets.push_back(PresetInfo("+proj=longlat +datum=WGS84", "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +nadgrids=" + dir + "BETA2007.gsb", "Gauss Krueger BW + Korrektur "));

    for (int i = 0; i < presets.size(); i++)
    {
        ui->presetBox->addItem(presets[i].name.c_str());
    }


    ui->XOffsetSpin->setDecimals(10);
    ui->XOffsetSpin->setMaximum(10000000);
    ui->XOffsetSpin->setMinimum(-10000000);
    //ui->XOffsetSpin->setValue(-5439122.807299255);
    //ui->XOffsetSpin->setValue(-926151);
    //ui->XOffsetSpin->setValue(-3506000);
    ui->XOffsetSpin->setValue(0);
    ui->YOffsetSpin->setDecimals(10);
    ui->YOffsetSpin->setMaximum(10000000);
    ui->YOffsetSpin->setMinimum(-10000000);
    //ui->YOffsetSpin->setValue(-984970.1841083583);
    //ui->YOffsetSpin->setValue(-3463995);
    //ui->YOffsetSpin->setValue(-5400147);
    ui->YOffsetSpin->setValue(0);
    ui->ZOffsetSpin->setDecimals(10);
    ui->ZOffsetSpin->setMaximum(10000000);
    ui->ZOffsetSpin->setMinimum(-10000000);
    //ui->ZOffsetSpin->setValue(-399.4944465);
    ui->ZOffsetSpin->setValue(0.0);
    //ui->FromDatumEdit->setText(QString("WGS84"));
    /*
#ifdef WIN32
	char *pValue;
	size_t len;
	errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
	if (err || pValue == NULL || strlen(pValue) == 0)
		err = _dupenv_s(&pValue, &len, "COVISEDIR");
	if (err)
		pValue="";
	QString covisedir = pValue;
#else
	QString covisedir = getenv("ODDLOTDIR");
	if (covisedir == "")
		covisedir = getenv("COVISEDIR");
#endif
    QString dir = covisedir + "/share/covise/";*/
    //ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam"));
    //ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +nadgrids="+dir+"BETA2007.gsb"));
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();
    //ui->ToProjectionEdit->setText(QString("utm +zone=50 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"));

    /*QString projFromString = "+proj=latlong +datum=" + ui->FromDatumEdit->text();
    //std::string projToString = "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47";
    QString projToString = "+proj=" + ui->ToProjectionEdit->text();
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();
    
    //TODO: substitute with projectData implementation!
    //Converts string representation of coordinate system into projPJ Object (pj_init_plus(...))
    if (!(new_pj_from = pj_init_plus(projFromString.toUtf8().constData())))
    {
        QMessageBox msg;
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection source: " + projFromString);
        msg.exec();
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << projFromString << std::endl;
        //return false;
    }
    else
    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
    {
        projectData_->setProj4ReferenceFrom(&new_pj_from);
        //pj_from = new_pj_from;
    }

    if (!(new_pj_to = pj_init_plus(projToString.toUtf8().constData())))
    {
        QMessageBox msg;
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection target: " + projToString);
        msg.exec();
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << projToString << std::endl;
        //return false;
    }
    else
    {
        projectData_->setProj4ReferenceTo(&new_pj_to);
        //pj_to = new_pj_to;
    }*/
    //inst = this;
}

ProjectionSettings::~ProjectionSettings()
{
    delete ui;
    delete projectData_;
}

void ProjectionSettings::transform(double &x, double &y, double &z)
{
    projPJ from = projectData_->getProj4ReferenceFrom();
    projPJ to = projectData_->getProj4ReferenceTo();
    if (from == nullptr|| to == nullptr)
    {
        updateSettings();
        if (from == nullptr || to == nullptr)
        {
            fprintf(stderr, "wrong prjection settings\n");
            return;
        }
    }
    int p = pj_transform(from,to,1,1,&x,&y,&z);
    if (p != 0)
    {
        fprintf(stderr, "pj_transform projection error %s\n", pj_strerrno(p));
    }
    x += XOffset;
    y += YOffset;
    z += ZOffset;
}

void ProjectionSettings::update()
{
    if(projectData_->getGeoReference() == nullptr) {
        ui->presetBox->setCurrentIndex(1);
    }
    else
    {
        //ui->ProjectionEdit->setText(projectData_->getGeoReference()->getParams());
        ui->ProjectionEdit->setText(pj_get_def(projectData_->getProj4ReferenceTo(),0));
        ui->SourceEdit->setText(pj_get_def(projectData_->getProj4ReferenceFrom(),0));
        updateUi();
    }
}

void ProjectionSettings::setProjectData(ProjectData *pd)
{
    projectData_ = pd;
    //connect(projectData_->getProjectWidget()->getMainWindow()->getFileSettings(), SIGNAL(accepted()), this, SLOT(okPressed()));
    update();
    updateSettings();
}

QString ProjectionSettings::prepareString(const QString &src)
{
    QString pre;
    if(src.indexOf("+init=epsg:") == -1)
    {
        pre = "+proj" + (src.section("+proj", 1,1));
    }
    else
    {
        pre = "+init=epsg:" + (src.section("+init=epsg:", 1,1)).section(' ',0,0);
    }
    return pre;
}

void ProjectionSettings::updateSettings()
{
    QMessageBox msg;
    QString projTo = ui->ProjectionEdit->text();
    QString projFrom = ui->SourceEdit->text();
    projPJ new_pj_from,new_pj_to;
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();

    new_pj_from = pj_init_plus((projFrom).toUtf8().constData());
    if(!(new_pj_from))
    {
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection source: " + projFrom);
        msg.exec();
    }
    else
    {
        new_pj_to = pj_init_plus((projTo).toUtf8().constData());
        projectData_->setProj4ReferenceFrom(new_pj_from);
        if(!(new_pj_to))
        {
            msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection target: " + projTo);
            msg.exec();
        }
        else
        {
            projectData_->setProj4ReferenceTo(new_pj_to);
            projectData_->setGeoReference(new GeoReference(projTo));
            updateUi();
        }
    }
}

void ProjectionSettings::checkProjForEPSG(const QString &proj)
{
    QString update;
    if(proj.indexOf("+init=epsg:") != -1) {
        update = proj.section("+init=epsg:",1,1).section(" ",0,0);
    }
    else
    {
        update = "None";
    }
    ui->EPSGEdit->setText(update);
}

void ProjectionSettings::checkProjForPreset(const QString &proj)
{
    //ui->presetBox->setCurrentIndex(presets.key(proj));
}

void ProjectionSettings::updateUi()
{
    //checkProjForEPSG(ui->Proj4Edit->text());
    //checkProjForPreset(ui->Proj4Edit->text());
}
