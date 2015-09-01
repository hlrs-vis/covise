#include "vrcwtrackinghw.h"

#include <QMessageBox>
#include <QStringBuilder>

#include "vrcwtrackingdim.h"
#include "vrcwhsensbindex.h"
#include "vrcwperson.h"
#include "datatypes.h"
#include "vrcwfinal.h"


/*****
 * constructor - destructor
 *****/

VRCWTrackingHw::VRCWTrackingHw(QWidget* parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI erstellen
   //
   //hand sensors
   handSensor_1 = new VRCWHSensBIndex(ui.handSensWidget_1);
   ui.handSensWidget_1HLayout->addWidget(handSensor_1);
   handSensor_1->setHSensLabel("Hand sensor 1");
   handSensor_2 = new VRCWHSensBIndex(ui.handSensWidget_2);
   ui.handSensWidget_2HLayout->addWidget(handSensor_2);
   handSensor_2->setHSensLabel("Hand sensor 2");
   handSensor_3 = new VRCWHSensBIndex(ui.handSensWidget_3);
   ui.handSensWidget_3HLayout->addWidget(handSensor_3);
   handSensor_3->setHSensLabel("Hand sensor 3");

   //head sensors
   headSensor_0 = new VRCWHSensBIndex(ui.headSensWidget_0);
   ui.headSensWidget_0HLayout->addWidget(headSensor_0);
   headSensor_0->setHSensLabel("Head sensor 0");
   headSensor_0->constHeaderDescOnly("Constant head - not tracked");
   headSensor_1 = new VRCWHSensBIndex(ui.headSensWidget_1);
   ui.headSensWidget_1HLayout->addWidget(headSensor_1);
   headSensor_1->setHSensLabel("Head sensor 1");
   headSensor_1->bodyIndexOnly();
   headSensor_1->headSensLayout();
   headSensor_2 = new VRCWHSensBIndex(ui.headSensWidget_2);
   ui.headSensWidget_2HLayout->addWidget(headSensor_2);
   headSensor_2->setHSensLabel("Head sensor 2");
   headSensor_2->bodyIndexOnly();
   headSensor_2->headSensLayout();
   headSensor_3 = new VRCWHSensBIndex(ui.headSensWidget_3);
   ui.headSensWidget_3HLayout->addWidget(headSensor_3);
   headSensor_3->setHSensLabel("Head sensor 3");
   headSensor_3->bodyIndexOnly();
   headSensor_3->headSensLayout();

   //persons
   person_1 = new VRCWPerson(ui.personWidget_1);
   ui.personWidget_1HLayout->addWidget(person_1);
   person_1->setPersonsLabel("Tracked person 1");
   person_2 = new VRCWPerson(ui.personWidget_2);
   ui.personWidget_2HLayout->addWidget(person_2);
   person_2->setPersonsLabel("Tracked person 2");
   person_3 = new VRCWPerson(ui.personWidget_3);
   ui.personWidget_3HLayout->addWidget(person_3);
   person_3->setPersonsLabel("Tracked person 3");
   person_4 = new VRCWPerson(ui.personWidget_4);
   ui.personWidget_4HLayout->addWidget(person_4);
   person_4->setPersonsLabel("Tracked person 4");
   person_5 = new VRCWPerson(ui.personWidget_5);
   ui.personWidget_5HLayout->addWidget(person_5);
   person_5->setPersonsLabel("Tracked person 5");
   person_6 = new VRCWPerson(ui.personWidget_6);
   ui.personWidget_6HLayout->addWidget(person_6);
   person_6->setPersonsLabel("Tracked person 6");
   person_7 = new VRCWPerson(ui.personWidget_7);
   ui.personWidget_7HLayout->addWidget(person_7);
   person_7->setPersonsLabel("Tracked person 7");
   person_8 = new VRCWPerson(ui.personWidget_8);
   ui.personWidget_8HLayout->addWidget(person_8);
   person_8->setPersonsLabel("Tracked person 8");
   person_9 = new VRCWPerson(ui.personWidget_9);
   ui.personWidget_9HLayout->addWidget(person_9);
   person_9->setPersonsLabel("Tracked person 9");


   //QVector<VRCWHSensBIndex*> handSensors erzeugen
   handSensors.append(handSensor_1);
   handSensors.append(handSensor_2);
   handSensors.append(handSensor_3);


   //QVector<VRCWHSensBIndex*> headSensors erzeugen
   headSensors.append(headSensor_0);
   headSensors.append(headSensor_1);
   headSensors.append(headSensor_2);
   headSensors.append(headSensor_3);


   //QVector<VRCWPerson*> persons erzeugen
   persons.append(person_1);
   persons.append(person_2);
   persons.append(person_3);
   persons.append(person_4);
   persons.append(person_5);
   persons.append(person_6);
   persons.append(person_7);
   persons.append(person_8);
   persons.append(person_9);


   //QVector<QWidget*> personWidgets erzeugen
   personWidgets.append(ui.personWidget_1);
   personWidgets.append(ui.personWidget_2);
   personWidgets.append(ui.personWidget_3);
   personWidgets.append(ui.personWidget_4);
   personWidgets.append(ui.personWidget_5);
   personWidgets.append(ui.personWidget_6);
   personWidgets.append(ui.personWidget_7);
   personWidgets.append(ui.personWidget_8);
   personWidgets.append(ui.personWidget_9);


   //Variable setzen
   expertMode = false;
   osData = Linux;


   //setup GUI
   ui.vrcTrackSysHintLabel->setWordWrap(true);
   ui.interSenseHintLabel->setWordWrap(true);
   setupGui();
   trackHandleParty_exec();
   numHandSens_exec(1);
   numHeadSens_exec(0);
   numPersons_exec(1);


   //Validator for Motionstar IP-Address
   QRegExp rxIp("((00[1-9]|0?[1-9][0-9]|[1][0-9][0-9]|2[0-4][0-9]|25[0-4])\\.)"
         "(([01]?[0-9]?[0-9]|2[0-4][0-9]|25[0-4])\\.){2}"
         "(00[1-9]|0?[1-9][0-9]|[1][0-9][0-9]|2[0-4][0-9]|25[0-4])");
   ui.moStarIPLineEdit->setValidator(new QRegExpValidator(rxIp,
         ui.moStarIPLineEdit));


   //Signals and Slots
   //
   connect(this, SIGNAL(trackSysChanged(trackSys)),
         handSensor_1, SLOT(trackSysChanged_exec(trackSys)));
   connect(this, SIGNAL(trackSysChanged(trackSys)),
         handSensor_2, SLOT(trackSysChanged_exec(trackSys)));
   connect(this, SIGNAL(trackSysChanged(trackSys)),
         handSensor_3, SLOT(trackSysChanged_exec(trackSys)));

   connect(this, SIGNAL(trackHandleVrcChecked(bool)),
         handSensor_1, SLOT(trackHandleVrcChecked_exec(bool)));
   connect(this, SIGNAL(trackHandleVrcChecked(bool)),
         handSensor_2, SLOT(trackHandleVrcChecked_exec(bool)));
   connect(this, SIGNAL(trackHandleVrcChecked(bool)),
         handSensor_3, SLOT(trackHandleVrcChecked_exec(bool)));
}


VRCWTrackingHw::~VRCWTrackingHw()
{

}


/*****
 * public functions
 *****/

// Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
// Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWTrackingHw::processGuiInput(const int& index,
      const QList<VRCWBase*>& vrcwList)
{
   const int WARN_01 = 1;
   const int WARN_02 = 2;
   const int WARN_03 = 3;
   const int WARN_04 = 4;
   const int WARN_05 = 5;
   const int WARN_06 = 6;
   const int WARN_07 = 7;
   const int WARN_08 = 8;
   const int WARN_09 = 9;
   const int ERROR_01 = 601;
   const int ERROR_02 = 602;
   const int ERROR_03 = 603;
   const int ERROR_04 = 604;
   const int ERROR_05 = 605;
   const int ERROR_06 = 606;
   const int ERROR_07 = 607;
   const int ERROR_08 = 608;
   const int ERROR_09 = 609;
   const int OK = 6;

   int success = DEF_ERROR;
   int awarning = DEF_WARN;


   //set variables
   //
   trackHandle trackingHandle = getGuiTrackHandle();
   trackSys trackingSys = getGuiTrackSys();
   bool checkArtHost = getGuiCheckArtHost();
   //hostToLookup = artHost || viconHost || moStarIP || 'empty string'
   QString hostToLookup;
   //hostIPLookupValue is used in a warning that can appear only
   // if trackingHandle == cov
   hIPLkpVal* hostIPLookupValue;
   //Polhemus and FOB
   QString sPortPolFob = getGuiPolFobSPort();
   //number of hand and head sensors
   int handSensNum = getGuiNumHandSens(); // int between 1 and 3
   int headSensNum = getGuiNumHeadSens(); //int between 0 and 3


   //
   //check values of body indexes of configured hand and head sensors
   //check optional values of button address if vrc
   //
   QList<int> bIndexes;

   //hand sensors
   for (int i = 0; i < handSensNum; ++i)
   {
      int hSensBIndex = handSensors[i]->getGuiBodyIndex();

      if (bIndexes.contains(hSensBIndex))
      {
         awarning = WARN_01;
         success = DEF_ERROR;
      }
      else
      {
         bIndexes.append(hSensBIndex);
         success = OK;
      }
   }

   //head sensors
   //for headSensNum == 0 nothing should happen
   for (int i = 1; i <= headSensNum; ++i)
   {
      int hSensBIndex = headSensors[i]->getGuiBodyIndex();

      if (bIndexes.contains(hSensBIndex))
      {
         awarning = WARN_01;
         success = DEF_ERROR;
      }
      else
      {
         bIndexes.append(hSensBIndex);
         success = OK;
      }
   }

   //hand sensor optional button address
   if (trackingHandle == vrc)
   {
      for (int i = 0; i < handSensNum; ++i)
      {
         switch (handSensors[i]->getGuiButtonSystem())
         {
            case ART_Fly://ART_Fly, Wand, Stylus and FOB_Mouse are the same
            case Wand:
            case Stylus:
            case FOB_Mouse:
            {
               //do nothing
               break;
            }
            case o_Optical: //o_Optical, Hornet, Mike and Wiimote are the same
            case Hornet:
            case Mike:
            case Wiimote:
            {
               int hSensVrcBAddr = handSensors[i]->getGuiVrcButtonAddr();

               if (bIndexes.contains(hSensVrcBAddr))
               {
                  awarning = WARN_02;
                  success = DEF_ERROR;
               }
               else
               {
                  bIndexes.append(hSensVrcBAddr);
                  success = OK;
               }
               break;
            }
         }
      }
   }


   //do the next checks only if tracking handled by COVISE
   //if tracking handled by VRC do nothing
   //
   if (trackingHandle == cov)
   {
      //
      //check values of ART host or IP, Vicon host or Motionstar IP
      //

      //host, ip einlesen in Abh. von tracksys
      //ueberpruefen ob gueltig
      //bei ART kann ein host, ip angegeben sein, bei Vicon und Motionstar
      //muss ein host oder ip angegeben sein

      bool checkHostIP = false;

      //check that the LineEdit isn't empty and set the text in the
      //lineEdit to the first entry
      switch (trackingSys)
      {
         case ART:
         {
            hostToLookup = getGuiArtHost();

            //the artHost must not be specified
            if (checkArtHost && !hostToLookup.isEmpty())
            {
               ui.artHostIPLineEdit->setText(hostToLookup);
               checkHostIP = true;
            }
            else
            {
               checkArtHost = false;
               ui.artHostIPCheckBox->setChecked(checkArtHost);
               ui.artHostIPLineEdit->clear();
            }
            break;
         }
         case Vicon:
         {
            hostToLookup = getGuiViconHost();

            if (!hostToLookup.isEmpty())
            {
               ui.viconHostIPLineEdit->setText(hostToLookup);
               checkHostIP = true;
            }
            else
            {
               awarning = WARN_03;
            }

            ui.viconHostIPLineEdit->setText(hostToLookup);
            break;
         }
         case Motionstar:
         {
            hostToLookup = getGuiMoStarIPAddr();
            QStringList ipParts = hostToLookup.split(
                  ".", QString::SkipEmptyParts);

            //es werden nur IP-Adressen mit vollstaendig gefuellten Feldern
            //akzeptiert
            //fuehrende Nullen werden entfernt
            if (ipParts.size() == 4)
            {
               hostToLookup.clear();
               for (QStringList::size_type i = 0; i < 4; ++i)
               {
                  for (int j = 0; j < ipParts[i].size(); ++j)
                  {
                     if ((ipParts[i].at(0) == '0') && (ipParts[i].size()) > 1)
                     {
                        ipParts[i].remove(0, 1);
                     }
                  }
                  hostToLookup.append(ipParts[i]);
                  if (i < 3)
                  {
                     hostToLookup.append(".");
                  }
               }

               ui.moStarIPLineEdit->setText(hostToLookup);
               checkHostIP = true;
            }
            else
            {
               awarning = WARN_04;
            }
            break;
         }
         default:
         {
            //do nothing
            break;
         }
      }

      //check the host or IP address
      if (checkHostIP)
      {
         if (expertMode)
         {
            //im expertMode wird der/die eingegebene Name/IP-Adresse
            //einfach ohne irgendwelche Ueberpruefung uebernommen
            success = OK;
         }
         else
         {
            hostIPLookupValue = hostIPLookup(hostToLookup);

            if (hostIPLookupValue->errCode == HIPLKP_EC_1)
            {
               awarning = WARN_05;
            }
            else
            {
               success = OK;
            }
         }
      }


      //
      //check values of FOB serial port and button device if not FOB Mouse
      //

      //check that the serial port is specified and if the lineEdit is visible
      //set the text in the lineEdit to the first entry
      switch (trackingSys)
      {
         case Polhemus://Polhemus and FOB are the same
         case FOB:
         {
            if (sPortPolFob.isEmpty())
            {
               awarning = WARN_06;
            }
            else if (ui.polFobSPortLineEdit->isVisible())
            {
               ui.polFobSPortLineEdit->setText(sPortPolFob);
               success = OK;
            }
            else
            {
               success = OK;
            }
            break;
         }
         default:
         {
            //do nothing
            break;
         }
      }


      //-check that the button devise is specified
      // and if the lineEdit is visible
      // set the text in the lineEdit to the first entry
      //-check that in case of FOB and Hornet/Mike
      // the specified serial ports are different
      //
      QList<QString> bDevList;

      //hand sensors
      for (int i = 0; i < handSensNum; ++i)
      {

         btnSys bSys = handSensors[i]->getGuiButtonSystem();

         switch (bSys)
         {
            case o_Optical://o_Optical, Hornet, Mike and Wiimote are the same
            case Hornet:
            case Mike:
            case Wiimote:
            {
               QString bDev = handSensors[i]->getGuiButtonDevice();

               if (bDevList.contains(bDev) || bDev.isEmpty())
               {
                  awarning = WARN_07;
                  success = DEF_ERROR;
               }
               else
               {
                  bDevList.append(bDev);
                  //set button device because in button device lineEdit
                  //we accept only the first entry
                  handSensors[i]->setButtonDevice(bDev);
                  success = OK;
               }

               //FOB serial port and hand devices serial ports
               if (trackingSys == FOB && awarning == DEF_WARN
                     && bDevList.contains(sPortPolFob))
               {
                  awarning = WARN_08;
               }

               break;
            }
            default:
            {
               //do nothing
               break;
            }
         }
      }
   }
   //end of checking if (trackingHandle == cov)


   //InterSense is only supported over the Polhemus Fastrak protocol
   switch (trackingSys)
   {
      case InterSense:
      {
         awarning = WARN_09;
         break;
      }
      default:
      {
         //do nothing
         break;
      }
   }



   //
   //show warnings
   //
   switch (awarning)
   {
      /*********
       * body index of one or more hand or head sensors are the same
       *********/
      case WARN_01:
      {
         QString message = tr("Some of the body indexes of the configured "
               "hand or head sensors are the same.\n\n"
               "Please define for every hand or head sensor a different "
               "body index.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_01;
         break;
      }
      /*********
       * body index and button address (if vrc) of one or more hand
       * or head sensors are the same
       *********/
      case WARN_02:
      {
         QString message = tr("Some of the body indexes or button addresses "
               "of the configured hand or head sensors are the same.\n\n"
               "Please define for every hand or head sensor a different "
               "body index and button address.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_02;
         break;
      }
     case WARN_03:
      {
         QString message = tr("You have not specified the host "
               "on which the Vicon tracking software is running.\n\n"
               "Please specify a hostname or IP address.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.viconHostIPLineEdit->setFocus();
         success = ERROR_03;
         break;
      }
      case WARN_04:
      {
         QString message = tr("You have not specified the IP address "
               "of the Motionstar Control-PC or the IP address isn't "
               "complete.\n\n"
               "Please specify a complete IP address.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.moStarIPLineEdit->setFocus();
         success = ERROR_04;
         break;
      }
      case WARN_05:
      {
         //prints for testing
         //
         qDebug() << "host: " << hostToLookup;
         qDebug() << "Lookup 1 failed:"
               << hostIPLookupValue->hostInfo_1.errorString();
         qDebug() << "Error 1: "
               << hostIPLookupValue->hostInfo_1.error();
         if (!hostIPLookupValue->hostInfo_2.errorString().isEmpty())
         {
            qDebug() << "Lookup 2 failed:"
                  << hostIPLookupValue->hostInfo_2.errorString();
            qDebug() << "Error 2: "
                  << hostIPLookupValue->hostInfo_2.error();
         }
         qDebug() << "";
         //


         QString message = tr("The hostname/IP address\n\n" "%1" "\n\n"
               "doesn't seem to be valid.\n"
               "Please make sure the hostname/IP address is in your "
               "hosts file\n"
               "or can be validated through a DNS lookup.")
               .arg(hostToLookup);
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         switch (trackingSys)
         {
            case ART:
            {
               ui.artHostIPLineEdit->selectAll();
               ui.artHostIPLineEdit->setFocus();
               break;
            }
            case Vicon:
            {
               ui.viconHostIPLineEdit->selectAll();
               ui.viconHostIPLineEdit->setFocus();
               break;
            }
            case Motionstar:
            {
               ui.moStarIPLineEdit->selectAll();
               ui.moStarIPLineEdit->setFocus();
               break;
            }
            default:
            {
               //do nothing
               break;
            }
         }

         success = ERROR_05;
         break;
      }
      case WARN_06:
      {
         QString polFobStr;
         switch (trackingSys)
         {
            case Polhemus:
            {
               polFobStr = trackSysToStr(Polhemus);
               break;
            }
            case FOB:
            {
               polFobStr = trackSysToStr(FOB);
               break;
            }
            default:
            {
               //do nothing
               break;
            }
         }
         QString message = tr("You have not specified the serial port "
               "on which the ") % polFobStr % tr(" is connected to COVISE "
               "GUI host.\n\n"
               "Please specify a serial port (serial/USB).");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.polFobSPortLineEdit->setFocus();
         success = ERROR_06;
         break;
      }
      case WARN_07:
      {
         QString message = tr("Some button devices use the same port/device "
               "or a port/device is not specified on which the button device "
               "is connected to the COVISE GUI host.\n\n"
               "Please use a different port/device on every button device.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_07;
         break;
      }
      case WARN_08:
      {
         QString message = tr("You have specified the same serial port "
               "for the FOB and a button device connection "
               "to the COVISE GUI host.\n\n"
               "Please specify different serial ports for the FOB "
               "and the button device connection.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_08;
         break;
      }
      case WARN_09:
      {
         QString message = tr("InterSense is only supported over "
               "the Polhemus Fastrak protocol.\n"
               "Please make sure, that your InterSense tracking system "
               "supports the Polhemus Fastrak protocol and that it "
               "is connected to a serial port.\n\n"
               "Choose Polhemus as the tracking system.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         ui.trackSysComboBox->setFocus();
         success = ERROR_09;
         break;
      }
      default:
      {
         qDebug() << "No warning appeared or warningCode can't be evaluated";
         break;
      }
   }



   //
   //sobald keine Fehler in der Eingabe sind:
   //- werden die Werte aus der GUI ausgelesen und in trackHwValDim gespeichert
   //- wird trackHwValDim an VRCWTrackingDim uebergeben und dort weiter bearbeitet
   //
   if (success == OK)
   {
      //Zeiger auf das VRCWTrackingDim-Widget erzeugen
      VRCWTrackingDim* trackingDim =
            dynamic_cast<VRCWTrackingDim*> (vrcwList[index + 1]);

      //Auslesen der GUI
      //
      int personsNum = getGuiNumPersons();
      trackHwValDim* thwvd = new trackHwValDim();
      thwvd->tHandle = trackingHandle;
      thwvd->vrcPort = getGuiVrcPort();
      thwvd->tSys = trackingSys;
      thwvd->checkArtHost = checkArtHost;
      thwvd->artHostSPort = getGuiArtHostSPort();
      thwvd->artRPort = getGuiArtRPort();
      thwvd->polFobSPort = sPortPolFob;
      // ART, Vicon, Motionstar host/IP Address stored in hostIPAddr
      thwvd->hostIPAddr = hostToLookup;
      thwvd->numHandSens = handSensNum;
      //thwvd->handSVal siehe unten
      thwvd->numHeadSens = headSensNum;
      //thwvd->headSVal siehe unten
      thwvd->numPersons = personsNum;
      //thwvd->persVal siehe unten

      //hand sensors in thwvd->handSVal
      for (int i = 0; i < handSensNum; ++i)
      {
         //Werte aus GUI in Klasse handSensVal speichern
         handSensVal* hSensVal = new handSensVal();

         hSensVal->sensLabel = handSensors[i]->getGuiHSensLabel();
         hSensVal->bSys = handSensors[i]->getGuiButtonSystem();
         hSensVal->bDev = handSensors[i]->getGuiButtonDevice();
         hSensVal->bDrv = handSensors[i]->getGuiButtonDriver();
         hSensVal->bIndex = handSensors[i]->getGuiBodyIndex();
         hSensVal->bAddr = handSensors[i]->getGuiVrcButtonAddr();

         //Werte aus Gui in Klasse handSensVal in Klasse trackHwValDim
         //in QVector speichern
         thwvd->handSVal.append(hSensVal);
      }

      //head sensors
      //for headSensNum == 0 is the constant head
      for (int i = 0; i <= headSensNum; ++i)
      {
         //Werte aus GUI in Klasse headSensVal speichern
         headSensVal* hSensVal = new headSensVal();

         hSensVal->sensLabel = headSensors[i]->getGuiHSensLabel();

         //headsensor_0 has no buttonIndex
         if (i > 0)
         {
            hSensVal->bIndex = headSensors[i]->getGuiBodyIndex();
         }

         //Werte aus Gui in Klasse headSensVal in Klasse trackHwValDim
         //in QVector speichern
         thwvd->headSVal.append(hSensVal);
      }

      //tracked persons
      for (int i = 0; i < personsNum; ++i)
      {
         personVal* pVal = new personVal();

         pVal = persons[i]->getGuiPerson();

         thwvd->persVal.append(pVal);
      }


      //Werte an vrcwtrackingDim uebergeben
      //VRCWTrackingHw uebergibt keine Werte mehr an vrcwfinal
      //thwvd wird in VRCWTrackingDim mit weiteren Werten gefuellt
      //und von dort an VRCWFinal uebergeben
      trackingDim->setTHwVDData(thwvd);
   }

   return success;
}

//set variable osData (operating system) and if changed execute setupGui()
//set OS in hand sensors
//
void VRCWTrackingHw::setOS_GUI(const opSys& os)
{
   if (osData != os)
   {
      osData = os;
      //execute setupGui after setting osData
      setupGui();
   }

   //set os in hand sensors
   for (int i = 0; i < handSensors.size(); ++i)
   {
      handSensors[i]->setOS_GUI(os);
   }
}


/*****
 * private functions
 *****/

// Auslesen des GUI
//
trackHandle VRCWTrackingHw::getGuiTrackHandle() const
{
   trackHandle tH;

   if (ui.trackHandleCoviseRadioButton->isChecked())
   {
      tH = cov;
   }
   else
   {
      tH = vrc;
   }

   return tH;
}

int VRCWTrackingHw::getGuiVrcPort() const
{
   return ui.vrcPortSpinBox->value();
}

trackSys VRCWTrackingHw::getGuiTrackSys() const
{
   return strToTrackSys(ui.trackSysComboBox->currentText());
}

bool VRCWTrackingHw::getGuiCheckArtHost() const
{
   return ui.artHostIPCheckBox->isChecked();
}

QString VRCWTrackingHw::getGuiArtHost() const
{
   QStringList artHostList = ui.artHostIPLineEdit->text().split(
         QRegExp("\\s+"), QString::SkipEmptyParts);

   //we only use the first entry
   if (artHostList.isEmpty())
   {
      artHostList.append("");
   }
   return artHostList[0].toLower();
}

int VRCWTrackingHw::getGuiArtHostSPort() const
{
   return ui.artHostIPPortSpinBox->value();
}

int VRCWTrackingHw::getGuiArtRPort() const
{
   return ui.artRecPortSpinBox->value();
}

QString VRCWTrackingHw::getGuiViconHost() const
{
   QStringList viconHostList = ui.viconHostIPLineEdit->text().split(
         QRegExp("\\s+"), QString::SkipEmptyParts);

   //we only use the first entry
   if (viconHostList.isEmpty())
   {
      viconHostList.append("");
   }
   return viconHostList[0].toLower();
}

QString VRCWTrackingHw::getGuiPolFobSPort() const
{
   QString sPort = ui.polFobSPortComboBox->currentText();

   if (sPort == OTHER)
   {
      QStringList sPortList = ui.polFobSPortLineEdit->text().split(
            QRegExp("\\s+"), QString::SkipEmptyParts);

      //we only use the first entry
      if (sPortList.isEmpty())
      {
         sPortList.append("");
      }
      sPort = sPortList[0];
   }

   return sPort;
}

QString VRCWTrackingHw::getGuiMoStarIPAddr() const
{
   return ui.moStarIPLineEdit->text();
}

int VRCWTrackingHw::getGuiNumHandSens() const
{
   return ui.handSensCountSpinBox->value();
}

int VRCWTrackingHw::getGuiNumHeadSens() const
{
   return ui.headSensCountSpinBox->value();
}

int VRCWTrackingHw::getGuiNumPersons() const
{
   return ui.personsCountSpinBox->value();
}


// Setzen des GUI
//
void VRCWTrackingHw::setPersonsHandSens(const int& numHS)
{
   QStringList hSens;

   for (int i = 1; i <= numHS; ++i)
   {
      QString hStr = "Hand sensor " + QString::number(i);
      hSens.append(hStr);
   }

   for (int i = 0 ; i < persons.size(); ++i)
   {
      persons[i]->setHandSensCBoxContent(hSens);
   }
}

void VRCWTrackingHw::setPersonsHeadSens(const int& numHS)
{
   QStringList hSens;

   for (int i = 0; i <= numHS; ++i)
   {
      QString hStr = "Head sensor " + QString::number(i);
      hSens.append(hStr);
   }

   for (int i = 0 ; i < persons.size(); ++i)
   {
      persons[i]->setHeadSensCBoxContent(hSens);
   }
}

//setup GUI depending on osData and/or vrc
//
void VRCWTrackingHw::setupGui() const
{
   //ButtonSystem will be set using a connection between
   //trackSysComboBox and trackSys_exec()
   //and then by a connection between
   //buttonSysCombobox and buttonSys_exec()

   //clear comboBoxes
   ui.trackSysComboBox->clear();
   ui.polFobSPortComboBox->clear();

   //set TrackingSystem Combobox for Linux and Windows
   ui.trackSysComboBox->addItem(trackSysToStr(ART));
   ui.trackSysComboBox->addItem(trackSysToStr(Vicon));
   ui.trackSysComboBox->addItem(trackSysToStr(InterSense));
   ui.trackSysComboBox->addItem(trackSysToStr(Polhemus));
   ui.trackSysComboBox->addItem(trackSysToStr(FOB));
   ui.trackSysComboBox->addItem(trackSysToStr(Motionstar));

   switch (osData)
   {
      case Linux:
      {
         //TrackingSystem Combobox
         //nothing specific

         //Polhemus-FOB-SerialPort Combobox
         ui.polFobSPortComboBox->addItem(OTHER);
         ui.polFobSPortComboBox->addItem("/dev/ttyS0");
         ui.polFobSPortComboBox->addItem("/dev/ttyS1");
         ui.polFobSPortComboBox->addItem("/dev/usb/ttyUSB0");
         ui.polFobSPortComboBox->addItem("/dev/usb/ttyUSB1");
         break;
      }
      case Windows:
      {
         //TrackingSystem Combobox
         if (getGuiTrackHandle() == vrc)
         {
            int index = ui.trackSysComboBox->findText(trackSysToStr(Vicon));
            if (index >= 0)
            {
               ui.trackSysComboBox->removeItem(index);
            }
            ui.trackSysComboBox->addItem(trackSysToStr(Wii));
         }

         //Polhemus-FOB-SerialPort Combobox
         ui.polFobSPortComboBox->addItem(OTHER);
         ui.polFobSPortComboBox->addItem("COM1");
         ui.polFobSPortComboBox->addItem("COM2");
         ui.polFobSPortComboBox->addItem("COM3");
         break;
      }
   }

   //set default index for comboBoxes
   ui.trackSysComboBox->setCurrentIndex(0);
   ui.polFobSPortComboBox->setCurrentIndex(1);
}


/*****
 * private slots
 *****/

//show/hide VRC Port spinbox
//show/hide trackSysAddsWidget
//setupGui
//
void VRCWTrackingHw::trackHandleParty_exec()
{
   bool vrcRBChecked = ui.trackHandleVrcRadioButton->isChecked();

   if (vrcRBChecked)
   {
      ui.vrcPortWidget->show();
      ui.vrcTrackSysHintWidget->show();
      ui.trackSysAddsWidget->hide();
   }
   else
   {
      ui.vrcPortWidget->hide();
      ui.vrcTrackSysHintWidget->hide();
      ui.trackSysAddsWidget->show();
   }

   //InterSense Hint is only available if Tracking System InterSense
   //is chosen
   ui.interSenseHintWidget->hide();

   setupGui();

   //emit/send signal
   emit trackHandleVrcChecked(vrcRBChecked);
}

//set button system depending on tracking system
//enable input for additional information about Tracking system
//
void VRCWTrackingHw::trackSys_exec()
{
   trackSys guiTrackSys = getGuiTrackSys();

   //input for additional information about Tracking system
   //set VRC server hint
   if (ui.trackHandleVrcRadioButton->isChecked())
   {
      ui.vrcTrackSysHintWidget->show();
      if (guiTrackSys == InterSense)
      {
         ui.vrcTrackSysHintWidget->hide();
      }
   }

   switch (guiTrackSys)
   {
      case ART:
      {
         //ART
         ui.artHostPortWidget->show();
         //other
         ui.viconHostIPWidget->hide();
         ui.interSenseHintWidget->hide();
         ui.polFobSPortWidget->hide();
         ui.moStarIPWidget->hide();
         //VRC server hint
         ui.vrcTrackSysHintLabel->setText("Start the ART_DTRACKserver "
               "in a console/terminal.");
         break;
      }
      case Vicon:
      {
         //Vicon
         ui.viconHostIPWidget->show();
         //other
         ui.artHostPortWidget->hide();
         ui.interSenseHintWidget->hide();
         ui.polFobSPortWidget->hide();
         ui.moStarIPWidget->hide();
         //VRC server hint
         ui.vrcTrackSysHintLabel->setText("Start the VICONserver "
               "in a console/terminal.");
         break;
      }
      case InterSense:
      {
         //InterSense
         ui.interSenseHintWidget->show();
         //other
         ui.artHostPortWidget->hide();
         ui.viconHostIPWidget->hide();
         ui.polFobSPortWidget->hide();
         ui.moStarIPWidget->hide();
         //VRC
         ui.vrcTrackSysHintLabel->clear();
         //InterSense hint
         ui.interSenseHintLabel->setText("InterSense is only supported "
               "over the Polhemus Fastrak protocol.\nPlease make sure, "
               "that your InterSense tracking system supports the "
               "Polhemus Fastrak protocol and that it is connected "
               "to a serial port.\n"
               "->Choose Polhemus as the Tracking System.");
         break;
      }
      case Polhemus://Polhemus and FOB are nearly the same
      case FOB:
      {
         //FOB
         ui.polFobSPortWidget->show();
         //other
         ui.artHostPortWidget->hide();
         ui.viconHostIPWidget->hide();
         ui.interSenseHintWidget->hide();
         ui.moStarIPWidget->hide();
         //VRC server hint
         if (guiTrackSys == Polhemus)
         {
            ui.vrcTrackSysHintLabel->setText("Start the POLHEMUSserver "
                  "in a console/terminal.");
         }
         else
         {
            ui.vrcTrackSysHintLabel->setText("Start the FOBserver "
               "in a console/terminal.");
         }
         break;
      }
      case Motionstar:
      {
         //Motionstar
         ui.moStarIPWidget->show();
         //other
         ui.artHostPortWidget->hide();
         ui.viconHostIPWidget->hide();
         ui.interSenseHintWidget->hide();
         ui.polFobSPortWidget->hide();
         //VRC server hint
         ui.vrcTrackSysHintLabel->setText("Start the MOTIONSTARserver "
               "in a console/terminal.");
         break;
      }
      case Wii:
      {
         //weiss nicht, was ich hier anzeigen lassen soll
         //man braucht hoechstens das Input Device
         //und das auch beim Button Device

         //other
         ui.artHostPortWidget->hide();
         ui.viconHostIPWidget->hide();
         ui.interSenseHintWidget->hide();
         ui.polFobSPortWidget->hide();
         ui.moStarIPWidget->hide();
         //VRC server hint (only available in Windows)
         ui.vrcTrackSysHintLabel->setText("Start WiiAutoConnect for "
               "auto connect the Wii with BlueSoleil Bluetooth.\n"
               "Or start 'WiiMotionPlus Server' for manual connection.");
         break;
      }
   }

   //send signal
   emit trackSysChanged(guiTrackSys);
}

//show/hide config option for hand sensors depending on number of sensors
//
void VRCWTrackingHw::numHandSens_exec(const int& numHS)
{
   switch (numHS)
   {
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
      default:
      {
         //do nothing
         break;
      }
   }

   //Persons Hand sensor ComboBox
   setPersonsHandSens(numHS);
}

//show/hide config option for head sensors depending on number of sensors
//
void VRCWTrackingHw::numHeadSens_exec(const int& numHS)
{
   switch (numHS)
   {
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
      default:
      {
         //do nothing
         break;
      }
   }

   //Persons Head Sensor ComboBox
   setPersonsHeadSens(numHS);
}

//show/hide config option for tracked persons depending
//on number of tracked persons
//
void VRCWTrackingHw::numPersons_exec(const int& numP) const
{
   //show person 1 hide other
   ui.personWidget_1->show();
   ui.personWidget_2->hide();
   ui.personWidget_3->hide();
   ui.personWidget_4->hide();
   ui.personWidget_5->hide();
   ui.personWidget_6->hide();
   ui.personWidget_7->hide();
   ui.personWidget_8->hide();
   ui.personWidget_9->hide();

   //Person 1 is always shown (ui.personWidget_1)
   //show widgets depending on numP
   for (int i = 2 ; i <= numP; ++i)
   {
      personWidgets[i - 1]->show();
   }
}

//enable/disable lineEdit and port
//
void VRCWTrackingHw::artHostIP_Exec(const bool& checked) const
{
   //checked == ui.artHostIPCheckBox->isChecked()
   if (checked)
   {
      ui.artHostIPLineEdit->setEnabled(true);
      ui.artHostIPPortLabel->setEnabled(true);
      ui.artHostIPPortSpinBox->setEnabled(true);
   }
   else
   {
      ui.artHostIPLineEdit->setEnabled(false);
      ui.artHostIPPortLabel->setEnabled(false);
      ui.artHostIPPortSpinBox->setEnabled(false);
   }
}

//show/hide FOB-SerialPort LineEdit
//
void VRCWTrackingHw::polFobSPort_exec(const QString& pfSP) const
{
   if (pfSP == OTHER)
   {
      ui.polFobSPortLineEdit->show();
   }
   else
   {
      ui.polFobSPortLineEdit->hide();
   }
}

//Setzen der Variable expertMode
//
void VRCWTrackingHw::setExpertMode(const bool& changed)
{
   expertMode = changed;
}
