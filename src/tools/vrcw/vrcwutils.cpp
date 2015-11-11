#include "vrcwutils.h"

#include <QHash>
#include <QHostInfo>
#include <QHostAddress>
#include <QStringList>

#include <iostream>

#include "datatypes.h"


/*****
 * functions
 *****/

//Umwandlung von QString in enum cWall
//
cWall strToCWall(const QString& cw)
{
   QHash<QString, cWall> strToCW;

   strToCW.insert("Front", Front);
   strToCW.insert("Left", Left);
   strToCW.insert("Right", Right);
   strToCW.insert("Bottom", Bottom);
   strToCW.insert("Top", Top);
   strToCW.insert("Back", Back);

   return strToCW.value(cw);
}

//Umwandlung von QString in enum proKind
//
proKind strToProKind(const QString& pk)
{
   QHash<QString, proKind> strToPK;

   strToPK.insert("Powerwall", Powerwall);
   strToPK.insert("CAVE", CAVE);
   strToPK.insert("3D TV", _3D_TV);

   return strToPK.value(pk);
}

//Umwandlung von QString in enum stType
//
stType strToStType(const QString& st)
{
   QHash<QString, stType> strToST;

   strToST.insert("passive", passive);
   strToST.insert("active", active);
   strToST.insert("top & bottom", topBottom);
   strToST.insert("side by side", sideBySide);
   strToST.insert("checkerboard", checkerboard);
   strToST.insert("vertical interlaced", vInterlaced);
   strToST.insert("column interleaving", cInterleave);
   strToST.insert("horizontal interlaced", hInterlaced);
   strToST.insert("row interleaving", rInterleave);

   return strToST.value(st);
}

//Umwandlung von QString in enum trackSys
//
trackSys strToTrackSys(const QString& ts)
{
   QHash<QString, trackSys> strToTS;

   strToTS.insert("ART", ART);
   strToTS.insert("Vicon", Vicon);
   strToTS.insert("InterSense", InterSense);
   strToTS.insert("Polhemus", Polhemus);
   strToTS.insert("FOB", FOB);
   strToTS.insert("Motionstar", Motionstar);
   strToTS.insert("Wii", Wii);

   return strToTS.value(ts);
}

//Umwandlung von QString in enum btnSys
//
btnSys strToBtnSys(const QString& bs)
{
   QHash<QString, btnSys> strToBS;

   strToBS.insert("ART Flystick", ART_Fly);
   strToBS.insert("other Optical", o_Optical);
   strToBS.insert("Wand", Wand);
   strToBS.insert("Stylus", Stylus);
   strToBS.insert("FOB Mouse", FOB_Mouse);
   strToBS.insert("Hornet", Hornet);
   strToBS.insert("Mike", Mike);
   strToBS.insert("Wiimote", Wiimote);

   return strToBS.value(bs);
}

//Umwandlung von QString in enum btnDrv
//
btnDrv strToBtnDrv(const QString& bd)
{
   QHash<QString, btnDrv> strToBD;

   strToBD.insert("Mouse Buttons", Mouse_Buttons);
   strToBD.insert("DTrack", DTrack);
   strToBD.insert("Polhemus Driver", PolhemusDrv);
   strToBD.insert("FOB Driver", FOB_Drv);
   strToBD.insert("Motionstar Driver", MotionstarDrv);
   strToBD.insert("Hornet Driver", HornetDrv);
   strToBD.insert("Mike Driver", MikeDrv);
   strToBD.insert("Wiimote Driver", WiimoteDrv);

   return strToBD.value(bd);
}

//Umwandlung von QString in enum opSys
//
opSys strToOpSys(const QString& os)
{
   QHash<QString, opSys> strToOS;

   strToOS.insert("Linux", Linux);
   strToOS.insert("Windows", Windows);

   return strToOS.value(os);
}

//Umwandlung von QString in enum execMode
//
execMode strToExecMode(const QString& em)
{
   QHash<QString, execMode> strToEM;

   strToEM.insert("ssh", ssh);
   strToEM.insert("covremote", covremote);
   strToEM.insert("CovDaemon", CovDaemon);
   strToEM.insert("rsh", rsh);

   return strToEM.value(em);
}

//Umwandlung von QString in enum typePro
//
typePro strToTypePro(const QString& tp)
{
   QHash<QString, typePro> strToTP;

   strToTP.insert("Projector", Projector);
   strToTP.insert("Monitor", Monitor);

   return strToTP.value(tp);
}

//Umwandlung von QString in enum zPoint
//
zPoint strToZeroPoint(const QString& zp)
{
   QHash<QString, zPoint> strToZP;

   strToZP.insert("Cave dimension", cDim);
   strToZP.insert("Front wall", fWall);
   strToZP.insert("Bottom wall", boWall);

   return strToZP.value(zp);
}


//Umwandlung von enum cWall in QString
//
QString cWallToStr(const cWall& cw)
{
   QHash<cWall, QString> cwToStr;

   cwToStr.insert(Front, "Front");
   cwToStr.insert(Left, "Left");
   cwToStr.insert(Right, "Right");
   cwToStr.insert(Bottom, "Bottom");
   cwToStr.insert(Top, "Top");
   cwToStr.insert(Back, "Back");

   return cwToStr.value(cw);
}

///Umwandlung von enum proKind in QString
//
QString proKindToStr(const proKind& pk)
{
   QHash<proKind, QString> pkToStr;

   pkToStr.insert(Powerwall, "Powerwall");
   pkToStr.insert(CAVE, "CAVE");
   pkToStr.insert(_3D_TV, "3D TV");

   return pkToStr.value(pk);
}

//Umwandlung von enum stType in QString
//
QString stTypeToStr(const stType& st)
{
   QHash<stType, QString> stToStr;

   stToStr.insert(passive, "passive");
   stToStr.insert(active, "active");
   stToStr.insert(topBottom, "top & bottom");
   stToStr.insert(sideBySide, "side by side");
   stToStr.insert(checkerboard, "checkerboard");
   stToStr.insert(vInterlaced, "vertical interlaced");
   stToStr.insert(cInterleave, "column interleaving");
   stToStr.insert(hInterlaced, "horizontal interlaced");
   stToStr.insert(rInterleave, "row interleaving");

   return stToStr.value(st);
}

//Umwandlung von enum trackSys in QString
//
QString trackSysToStr(const trackSys& ts)
{
   QHash<trackSys, QString> tsToStr;

   tsToStr.insert(ART, "ART");
   tsToStr.insert(Vicon, "Vicon");
   tsToStr.insert(InterSense, "InterSense");
   tsToStr.insert(Polhemus, "Polhemus");
   tsToStr.insert(FOB, "FOB");
   tsToStr.insert(Motionstar, "Motionstar");
   tsToStr.insert(Wii, "Wii");

   return tsToStr.value(ts);
}

//Umwandlung von enum btnSys in QString
//
QString btnSysToStr(const btnSys& bs)
{
   QHash<btnSys, QString> bsToStr;

   bsToStr.insert(ART_Fly, "ART Flystick");
   bsToStr.insert(o_Optical, "other Optical");
   bsToStr.insert(Wand, "Wand");
   bsToStr.insert(Stylus, "Stylus");
   bsToStr.insert(FOB_Mouse, "FOB Mouse");
   bsToStr.insert(Hornet, "Hornet");
   bsToStr.insert(Mike, "Mike");
   bsToStr.insert(Wiimote, "Wiimote");

   return bsToStr.value(bs);
}

//Umwandlung von enum btnDrv in QString
//
QString btnDrvToStr(const btnDrv& bd)
{
   QHash<btnDrv, QString> bdToStr;

   bdToStr.insert(Mouse_Buttons, "Mouse Buttons");
   bdToStr.insert(DTrack, "DTrack");
   bdToStr.insert(PolhemusDrv, "Polhemus Driver");
   bdToStr.insert(FOB_Drv, "FOB Driver");
   bdToStr.insert(MotionstarDrv, "Motionstar Driver");
   bdToStr.insert(HornetDrv, "Hornet Driver");
   bdToStr.insert(MikeDrv, "Mike Driver");
   bdToStr.insert(WiimoteDrv, "Wiimote Driver");

   return bdToStr.value(bd);
}

//Umwandlung von enum opSys in QString
//
QString opSysToStr(const opSys& os)
{
   QHash<opSys, QString> osToStr;

   osToStr.insert(Linux, "Linux");
   osToStr.insert(Windows, "Windows");

   return osToStr.value(os);
}

//Umwandlung von enum execMode in QString
//
QString execModeToStr(const execMode& em)
{
   QHash<execMode, QString> emToStr;

   emToStr.insert(ssh, "ssh");
   emToStr.insert(covremote, "covremote");
   emToStr.insert(CovDaemon, "CovDaemon");
   emToStr.insert(rsh, "rsh");

   return emToStr.value(em);
}

//Umwandlung von enum typePro in QString
//
QString typeProToStr(const typePro& tp)
{
   QHash<typePro, QString> tpToStr;

   tpToStr.insert(Projector, "Projector");
   tpToStr.insert(Monitor, "Monitor");

   return tpToStr.value(tp);
}

//Umwandlung von enum zPoint in QString
//
QString zeroPointToStr(const zPoint& zp)
{
   QHash<zPoint, QString> zpToStr;

   zpToStr.insert(cDim, "Cave dimension");
   zpToStr.insert(fWall, "Front wall");
   zpToStr.insert(boWall, "Bottom wall");

   return zpToStr.value(zp);
}


//for validation of given hostname or IP-Address
//
hIPLkpVal* hostIPLookup(const QString& host)
{
   //_Voraussetzung:_
   //jeder host hat eine IP
   //max. einen/keinen FQDN
   //max. einen/keinen shortname/alias/hostname == FQDN ohne Domaenenanteil

   //mit hostname einen lookup machen -> IP-Adresse
   //mit IP-Adresse eine reverseLookup machen -> FQDN, falls vorhanden
   //bei FQDN den Domaenenanteil abschneiden und mit hostname
   // vergleichen
   //den gefundenen Namen mit der gefundenen IP-Adresse vergleichen
   // und damit sicherstellen, dass der hostname auch existiert
   // z.B. unter linux ergibt ein lookup mit 1111
   // ( QHostInfo::fromName(1111) ) eine IP-Adresse
   //den hostname, den FQDN und die IP-Adresse mit hosts vergleichen


   //hostname, FQDN or IP address for the host name lookup
   QString hostToLookup = host.toLower();

   //errorCode is used for different error messages or hints
   //if no error occured, errorCode is an empty QString
   QString errorCode;

   //do a host name lookup -> IP-Adresse
   QHostInfo hostInfo_1 = QHostInfo::fromName(hostToLookup);
   QString hostToLookupAddress_1;

   //for a IP lookup -> FQDN, if existing
   QHostInfo hostInfo_2;
   hostInfo_2.setErrorString("");
   QString hostToLookupName_2;

   //hostname from a FQDN from hostInfo_2
   QString hostname_2;


   if (hostInfo_1.error() != QHostInfo::NoError)
   {
      errorCode = HIPLKP_EC_1;
   }
   else
   {
      //foreach ist fuer hosts, die mehrere Adressen haben,
      //wie z.B. www.google.de
      std::cout << "Nur erste IP-Adresse wurde verwendet!" << std::endl;

      foreach (QHostAddress address, hostInfo_1.addresses())
      {
         qDebug() << "Error: " << hostInfo_1.error();
         std::cout << "host: " << qPrintable(hostToLookup) << std::endl;
         std::cout << "Lookup output: "
               << qPrintable(hostInfo_1.errorString()) << std::endl;
         std::cout << "Found address: "
               << qPrintable(address.toString()) << std::endl;
         std::cout << "Found name: " << qPrintable(hostInfo_1.hostName())
               << std::endl;
         std::cout << "" << std::endl;
      }
      //


      //IP address of hostToLookup
      hostToLookupAddress_1 = hostInfo_1.addresses().first().toString();

      //do a IP lookup
      //(mit IP-Adresse einen reverseLookup machen
      // -> FQDN, falls vorhanden)
      hostInfo_2 = QHostInfo::fromName(hostToLookupAddress_1);

      if (hostInfo_2.error() != QHostInfo::NoError)
      {
         errorCode = HIPLKP_EC_1;
      }
      else
      {
         //foreach ist fuer hosts, die mehrere Adressen haben,
         //wie z.B. www.google.de
         std::cout << "Reverse Lookup." << std::endl;
         std::cout << "Nur erste IP-Adresse wurde verwendet!"
               << std::endl;

         foreach (QHostAddress address, hostInfo_2.addresses())
         {
            qDebug() << "Error: " << hostInfo_2.error();
            std::cout << "host: " << qPrintable(hostToLookup)
                  << std::endl;
            std::cout << "Lookup output: "
                  << qPrintable(hostInfo_2.errorString()) << std::endl;
            std::cout << "Found address: "
                  << qPrintable(address.toString()) << std::endl;
            std::cout << "Found name: "
                  << qPrintable(hostInfo_2.hostName())
                  << std::endl;
            std::cout << "" << std::endl;
         }
         //


         //-> FQDN if available or hostname/name
         hostToLookupName_2 = hostInfo_2.hostName();

         //compare the found hostname/FQDN with the found IP address
         // to be sure, that the hostname/FQDN exist.
         // For example under linux a lookup with 1111
         // ( QHostInfo::fromName(1111) ) results in a IP address
         if (hostToLookupName_2 == hostToLookupAddress_1)
         {
            errorCode = HIPLKP_EC_1;
         }
         else
         {
            QStringList fqdnParts = hostToLookupName_2.split(".");
            hostname_2 = fqdnParts.first();
         }
      }
   }

   hIPLkpVal* hIPLkpValue = new hIPLkpVal();
   hIPLkpValue->toLookup = hostToLookup;
   hIPLkpValue->ipAddr_1 = hostToLookupAddress_1;
   hIPLkpValue->fqdn_2 = hostToLookupName_2;
   hIPLkpValue->hostname_2 = hostname_2;
   hIPLkpValue->errCode = errorCode;

   //for testing output
   hIPLkpValue->hostInfo_1 = hostInfo_1;
   hIPLkpValue->hostInfo_2 = hostInfo_2;

   return hIPLkpValue;
}
