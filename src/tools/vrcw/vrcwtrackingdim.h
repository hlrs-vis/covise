#ifndef VRCWTRACKINGDIM_H
#define VRCWTRACKINGDIM_H

#include <QWidget>
#include "ui_vrcwtrackingdim.h"

#include "vrcwbase.h"
#include "vrcwutils.h"

class VRCWSensorTrackSysDim;
class trackHwValDim;


class VRCWTrackingDim : public QWidget, public VRCWBase
{
   Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWTrackingDim(QWidget *parent = 0);
   ~VRCWTrackingDim();


   /*****
    * functions
    *****/
   // Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
   // Bearbeitung und Ueberpruefung der Eingaben im GUI
   int processGuiInput(const QList<VRCWBase*>& vrcwList);

   //set thwvdData and setup GUI
   void setTHwVDData(trackHwValDim*& thwvd);

   //set head sensor 0 (ConstHead) Offset with vPos
   //from dimPowerwall or dimCave
   void setCHeadOffset(QVector<int>& vPos);


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWTrackingDimClass ui;


   /*****
    * functions
    *****/
   // setup GUI
   void setupGui();

   // Auslesen des GUI



   /*****
    * variables
    *****/
   VRCWSensorTrackSysDim* trackingSystem;
   VRCWSensorTrackSysDim* handSensor_1;
   VRCWSensorTrackSysDim* handSensor_2;
   VRCWSensorTrackSysDim* handSensor_3;
   VRCWSensorTrackSysDim* headSensor_0;
   VRCWSensorTrackSysDim* headSensor_1;
   VRCWSensorTrackSysDim* headSensor_2;
   VRCWSensorTrackSysDim* headSensor_3;
   QVector<VRCWSensorTrackSysDim*> handSensors;
   QVector<VRCWSensorTrackSysDim*> headSensors;
   trackHwValDim* thwvdData;
   //store old values
   trackSys tSysOld;
   QVector<btnSys> bSysOld;

};

#endif // VRCWTRACKINGDIM_H
