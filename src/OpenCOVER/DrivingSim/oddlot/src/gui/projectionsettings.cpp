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
#include "ui_projectionsettings.h"


// Data //

ProjectionSettings *ProjectionSettings::inst = NULL;

void ProjectionSettings::okPressed()
{
    QString projFromString = "+proj=latlong +datum=" + ui->FromDatumEdit->text();
    //std::string projToString = "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47";.
    QString projToString = "+proj=" + ui->ToProjectionEdit->text();
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();

    if (!(pj_from = pj_init_plus(projFromString.toUtf8().constData())))
    {
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << projFromString << std::endl;
        //return false;
    }
    if (!(pj_to = pj_init_plus(projToString.toUtf8().constData())))
    {
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << projToString << std::endl;
        //return false;
    }
}
//################//
// CONSTRUCTOR    //
//################//

ProjectionSettings::ProjectionSettings()
    : ui(new Ui::ProjectionSettings)
{
    inst = this;
    ui->setupUi(this);

    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));

    ui->XOffsetSpin->setDecimals(10);
    ui->XOffsetSpin->setMaximum(10000000);
    ui->XOffsetSpin->setMinimum(-10000000);
    //ui->XOffsetSpin->setValue(-5439122.807299255);
    ui->XOffsetSpin->setValue(-3506000);
    //ui->XOffsetSpin->setValue(-926151);
    ui->YOffsetSpin->setDecimals(10);
    ui->YOffsetSpin->setMaximum(10000000);
    ui->YOffsetSpin->setMinimum(-10000000);
    //ui->YOffsetSpin->setValue(-984970.1841083583);
    //ui->YOffsetSpin->setValue(-3463995);
    ui->YOffsetSpin->setValue(-5400147);
    ui->ZOffsetSpin->setDecimals(10);
    ui->ZOffsetSpin->setMaximum(10000000);
    ui->ZOffsetSpin->setMinimum(-10000000);
    //ui->ZOffsetSpin->setValue(-399.4944465);
    ui->ZOffsetSpin->setValue(0.0);
    ui->FromDatumEdit->setText(QString("WGS84"));
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
	QString dir = covisedir + "/share/covise/";
    //ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam"));
    ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +nadgrids="+dir+"BETA2007.gsb"));
    //ui->ToProjectionEdit->setText(QString("utm +zone=50 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"));
    QString projFromString = "+proj=latlong +datum=" + ui->FromDatumEdit->text();
    //std::string projToString = "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47";.
    QString projToString = "+proj=" + ui->ToProjectionEdit->text();
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();
    
    projPJ new_pj_from, new_pj_to;
    if (!(new_pj_from = pj_init_plus(projFromString.toUtf8().constData())))
    {
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << projFromString << std::endl;
        //return false;
    }
    else
    {
        pj_from = new_pj_from;
    }
    if (!(new_pj_to = pj_init_plus(projToString.toUtf8().constData())))
    {
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << projToString << std::endl;
        //return false;
    }
    else
    {
        pj_to = new_pj_to;
    }
    inst = this;
}

ProjectionSettings::~ProjectionSettings()
{
    delete ui;
}

void ProjectionSettings::transform(double &x, double &y, double &z)
{
    pj_transform(pj_from, pj_to, 1, 1, &x, &y, &z);
    x += XOffset;
    y += YOffset;
    z += ZOffset;
}
