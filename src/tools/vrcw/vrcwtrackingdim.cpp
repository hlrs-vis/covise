#include "vrcwtrackingdim.h"

#include "vrcwsensortracksysdim.h"
#include "datatypes.h"
#include "vrcwfinal.h"


/*****
 * constructor - destructor
 *****/

VRCWTrackingDim::VRCWTrackingDim(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI erstellen
   //
   //Tracking System
   trackingSystem = new VRCWSensorTrackSysDim(ui.trackSysWidget);
   ui.trackSysWidgetVLayout->addWidget(trackingSystem);
   trackingSystem->setSensTrackSysLabel("Tracking system");

   //Hand Sensors
   handSensor_1 = new VRCWSensorTrackSysDim(ui.handSensWidget_1);
   ui.handSensWidget_1VLayout->addWidget(handSensor_1);
   handSensor_1->setSensTrackSysLabel("Hand sensor 1");
   handSensor_2 = new VRCWSensorTrackSysDim(ui.handSensWidget_2);
   ui.handSensWidget_2VLayout->addWidget(handSensor_2);
   handSensor_2->setSensTrackSysLabel("Hand sensor 2");
   handSensor_3 = new VRCWSensorTrackSysDim(ui.handSensWidget_3);
   ui.handSensWidget_3VLayout->addWidget(handSensor_3);
   handSensor_3->setSensTrackSysLabel("Hand sensor 3");

   //Const Head Sensor
   headSensor_0 = new VRCWSensorTrackSysDim(ui.headSensWidget_0);
   ui.headSensWidget_0VLayout->addWidget(headSensor_0);
   headSensor_0->setSensTrackSysLabel("Head sensor 0");
   headSensor_0->setSensTrackSysDesc("(Constant head - not tracked)");

   //Head Sensors
   headSensor_1 = new VRCWSensorTrackSysDim(ui.headSensWidget_1);
   ui.headSensWidget_1VLayout->addWidget(headSensor_1);
   headSensor_1->setSensTrackSysLabel("Head sensor 1");
   headSensor_1->hideSensTrackSysDesc();
   headSensor_2 = new VRCWSensorTrackSysDim(ui.headSensWidget_2);
   ui.headSensWidget_2VLayout->addWidget(headSensor_2);
   headSensor_2->setSensTrackSysLabel("Head sensor 2");
   headSensor_2->hideSensTrackSysDesc();
   headSensor_3 = new VRCWSensorTrackSysDim(ui.headSensWidget_3);
   ui.headSensWidget_3VLayout->addWidget(headSensor_3);
   headSensor_3->setSensTrackSysLabel("Head sensor 3");
   headSensor_3->hideSensTrackSysDesc();


   //QVector<VRCWHSensBIndex*> handSensors erzeugen
   handSensors.append(handSensor_1);
   handSensors.append(handSensor_2);
   handSensors.append(handSensor_3);

   //QVector<VRCWHSensBIndex*> headSensors erzeugen
   headSensors.append(headSensor_0);
   headSensors.append(headSensor_1);
   headSensors.append(headSensor_2);
   headSensors.append(headSensor_3);


   //Default value headsensor_0 Const head
   sensTrackSysDim* stsDim = new sensTrackSysDim();
   stsDim->x = 0;
   stsDim->y = -2000;
   stsDim->z = 0;
   headSensor_0->setSensTrackSysOffset(stsDim);
   delete stsDim;


   //Set Variables
   //
   thwvdData = new trackHwValDim();
   tSysOld = ART;
   //ein Vektor mit 3 Elementen, da max 3 handSensors konfiguriert werden koennen
   bSysOld = QVector<btnSys>(3);


   //setup GUI
   setupGui();
}

VRCWTrackingDim::~VRCWTrackingDim()
{

}

/*****
 * public functions
 *****/

// Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
// Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWTrackingDim::processGuiInput(
      const QList<VRCWBase*>& vrcwList)
{
   const int OK = 7;

   //int success = DEF_ERROR;

   //bis jetzt wird hier jede Eingabe akzeptiert und nicht ueberprueft
   int success = OK;

   //wird bei Anzeige dieser Klasse auf "Next>" gedrueckt:
   //(sobald keine Fehler in der Eingabe sind:)
   //- werden fuer das Tracking System und die definierten Sensoren die Werte
   //  ausgelesen und an VRCWFinal uebergeben
   //- wird in VRCWFinal die Generierung der config.xml angestossen
   //
   if (success == OK)
   {
      //GUI auslesen
      //
      trackHwValDim* thwvdDim = thwvdData;
      int handSensNum = thwvdData->numHandSens;
      int headSensNum = thwvdData->numHeadSens;

      //Tracking System
      //
      thwvdDim->tSysDim = trackingSystem->getGuiSensTrackSysDim();

      //hand sensors in thwvdDim->handSDim
      //
      for (int i = 0; i < handSensNum; ++i)
      {
         //Werte aus GUI in Klasse sensTrackSysDim speichern
         sensTrackSysDim* sTsDim = handSensors[i]->getGuiSensTrackSysDim();

         //Werte aus Gui in Klasse sensTrackSysDim in Klasse trackHwValDim
         //in QVector speichern
         if (thwvdDim->handSDim.size() == i)
         {
            thwvdDim->handSDim.append(sTsDim);
         }
         else
         {
            thwvdDim->handSDim[i] = sTsDim;
         }
      }

      //head sensors in thwvdDim->headSDim
      //
      for (int i = 0; i <= headSensNum; ++i)
      {
         //Werte aus GUI in Klasse sensTrackSysDim speichern
         sensTrackSysDim* sTsDim = headSensors[i]->getGuiSensTrackSysDim();

         //Werte aus Gui in Klasse sensTrackSysDim in Klasse trackHwValDim
         //in QVector speichern
         if (thwvdDim->headSDim.size() == i)
         {
            thwvdDim->headSDim.append(sTsDim);
         }
         else
         {
            thwvdDim->headSDim[i] = sTsDim;
         }
      }

      //Uebergabe der Werte an VRCWFinal
      int finalIndex = vrcwList.size() - 2;
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      final->setTrackingValDim(thwvdDim);

      //Generierung der Configs anstossen
      final->createXmlConfigs();
   }


   return success;
}

//get trackHwValDim from VRCWTrackingHw and setup GUI
//
void VRCWTrackingDim::setTHwVDData(trackHwValDim*& thwvd)
{
   thwvdData = thwvd;

   setupGui();
}

//set head sensor 0 (ConstHead) Offset with vPos
//from dimPowerwall or dimCave
//only if vPos is unequal zero
//
void VRCWTrackingDim::setCHeadOffset(QVector<int>& vPos)
{
    QVector<int> zeroVec(3,0);

    if (vPos != zeroVec)
    {
        sensTrackSysDim* stsDim = new sensTrackSysDim();
        stsDim->x = vPos[0];
        stsDim->y = vPos[1];
        stsDim->z = vPos[2];

        headSensor_0->setSensTrackSysOffset(stsDim);
    }
}



/*****
 * private functions
 *****/

//setup GUI
//
void VRCWTrackingDim::setupGui()
{
   int handSensNum = thwvdData->numHandSens;
   int headSensNum = thwvdData->numHeadSens;

   //
   //tracking system
   //

   trackSys tSysData = thwvdData->tSys;

   //Description
   //
   QString trackSysDesc = "(" + trackSysToStr(tSysData) + ")";
   trackingSystem->setSensTrackSysDesc(trackSysDesc);

   //Auswerten des TrackingSys -> setzen der transmitter orientation
   //nur wenn sich das TrackingSys geaendert hat
   //
   if (tSysData != tSysOld)
   {
      sensTrackSysDim* trackSysOrient = new sensTrackSysDim();

      switch (thwvdData->tSys)
      {
         case FOB://FOB and Motionstar are the same
         case Motionstar:
         {
            trackSysOrient->h = 90;
            trackSysOrient->p = 0;
            trackSysOrient->r = 180;
            break;
         }
         case InterSense://InterSense and Polhemus are the same
         case Polhemus:
         {
            trackSysOrient->h = 0;
            trackSysOrient->p = 180;
            trackSysOrient->r = 0;
            break;
         }
         case ART://ART, Vicon and Wii are the same
         case Vicon:
         case Wii:
         {
            //do nothing
            break;
         }
      }

      tSysOld = tSysData;
      trackingSystem->setSensTrackSysOrient(trackSysOrient);
   }


   //
   //hand sensors
   //

   //hand sensors in thwvdData->handSVal
   for (int i = 0; i < handSensNum; ++i)
   {
      //Description
      //

      //HandSensor Daten
      handSensVal* hsv = thwvdData->handSVal[i];
      btnSys bSysData = hsv->bSys;
      QString bSysString = btnSysToStr(hsv->bSys);
      QString bDevString = hsv->bDev;
      QString stsDesc = "(" + bSysString;

      if (bDevString.isEmpty())
      {
         stsDesc = stsDesc + ")";
      }
      else
      {
         stsDesc = stsDesc + " - " + bDevString + ")";
      }

      handSensors[i]->setSensTrackSysDesc(stsDesc);


      //Auswerten des ButtonSys -> setzen der hand orientation
      //nur wenn sich das ButtonSys geaendert hat
      //
      if (bSysData != bSysOld[i])
      {
         sensTrackSysDim* handSensOrient = new sensTrackSysDim();

         switch (bSysData)
         {
            case Wand:
            {
               handSensOrient->h = -90;
               handSensOrient->p = 0;
               handSensOrient->r = 180;
               break;
            }
            case FOB_Mouse:
            {
               handSensOrient->h = 90;
               handSensOrient->p = 0;
               handSensOrient->r = 180;
               break;
            }
            case Hornet://Hornet and Mike are the same
            case Mike:
            {
               handSensOrient->h = -90;
               handSensOrient->p = 0;
               handSensOrient->r = 0;
               break;
            }
            case ART_Fly://ART_Fly, o_Optical, Stylus and Wiimote are the same
            case o_Optical:
            case Stylus:
            {
               handSensOrient->h = 0;
               handSensOrient->p = 0;
               handSensOrient->r = 0;
               break;
            }
            case Wiimote:
            {
               switch (tSysData)
               {
                  case FOB://FOB and Motionstar are the same
                  case Motionstar:
                  {
                     handSensOrient->h = -90;
                     handSensOrient->p = 90;
                     handSensOrient->r = 0;
                     break;
                  }
                  case ART://ART, Vicon, InterSense, Polhemus and Wii are the same
                  case Vicon:
                  case InterSense:
                  case Polhemus:
                  case Wii:
                  {
                     handSensOrient->h = 0;
                     handSensOrient->p = 0;
                     handSensOrient->r = 0;
                     break;
                  }
               }
               break;
            }
         }

         bSysOld[i] = bSysData;
         handSensors[i]->setSensTrackSysOrient(handSensOrient);
      }
   }


   //show/hide der Widgets
   //
   switch (handSensNum)
   {
      default: //default should be case 1:
      case 1:
      {
         ui.handSensWidget_1->show();
         ui.handSensWidget_2->hide();
         ui.handSensWidget_3->hide();
         break;
      }
      case 2:
      {
         ui.handSensWidget_1->show();
         ui.handSensWidget_2->show();
         ui.handSensWidget_3->hide();
         break;
      }
      case 3:
      {
         ui.handSensWidget_1->show();
         ui.handSensWidget_2->show();
         ui.handSensWidget_3->show();
         break;
      }
   }



   //
   //head sensors
   //

   //headSensor_0 is always visible and name and description do not change

   //show/hide der Widgets
   //
   switch (headSensNum)
   {
      default: //default should be case 0:
      case 0:
      {
         ui.headSensWidget_1->hide();
         ui.headSensWidget_2->hide();
         ui.headSensWidget_3->hide();
         break;
      }
      case 1:
      {
         ui.headSensWidget_1->show();
         ui.headSensWidget_2->hide();
         ui.headSensWidget_3->hide();
         break;
      }
      case 2:
      {
         ui.headSensWidget_1->show();
         ui.headSensWidget_2->show();
         ui.headSensWidget_3->hide();
         break;
      }
      case 3:
      {
         ui.headSensWidget_1->show();
         ui.headSensWidget_2->show();
         ui.headSensWidget_3->show();
         break;
      }
   }
}

// Auslesen des GUI
//
