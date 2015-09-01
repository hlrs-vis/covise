#include "vrcwprojectiondimpowerwall.h"

#include <QMessageBox>
#include <QStringBuilder>
#include <QDebug>

#include "vrcwhost.h"
#include "vrcwhostprojection.h"
#include "vrcwtrackingdim.h"
#include "vrcwfinal.h"
#include "vrcwprojectionressizetiled.h"
#include "vrcwprojectionressize.h"
#include "vrcwprojectionvposfloor.h"
#include "datatypes.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionDimPowerwall::VRCWProjectionDimPowerwall(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI erstellen
   //
   projectionResSizeTiled = new VRCWProjectionResSizeTiled(ui.dimTiledWidget);
   ui.dimTiledWidgetVLayout->addWidget(projectionResSizeTiled);
   ui.dimTiledWidget->hide();
   projectionResSize = new VRCWProjectionResSize(ui.dimWidget);
   ui.dimWidgetVLayout->addWidget(projectionResSize);
   projectionVposFloor = new VRCWProjectionVposFloor(ui.posWidget);
   ui.posWidgetVLayout->addWidget(projectionVposFloor);

   //Variablen setzen
   kindData = Powerwall;
   tiledData = false;
   typePData = Projector;
   stereoData = passive;
   bothEyesData = false;
   grakaData = 0;
}

VRCWProjectionDimPowerwall::~VRCWProjectionDimPowerwall()
{

}


/*****
 * public functions
 *****/

//Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
//Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWProjectionDimPowerwall::processGuiInput(const int& index,
      const QList<VRCWBase*>& vrcwList)
{
   const int WARN_01 = 1;
   const int WARN_02 = 2;
   const int ERROR_01 = 301;
   const int ERROR_02 = 302;
   const int OK = 3;//same as in VRCWProjectionDimCave

   int success = DEF_ERROR;
   int awarning = DEF_WARN;

   //
   //doing some checks
   //

   //set variables
   QList<wallVal*> wallDims = getGuiPWallDim();
   int screenWidth = wallDims[0]->screenSize[0];
   int screenHeight = wallDims[0]->screenSize[1];
   int wallWidth = wallDims[0]->wallSize[0];
   int wallHeight = wallDims[0]->wallSize[1];

   if (screenWidth == 0 || screenHeight == 0)
   {
      awarning = WARN_01;
      success = DEF_ERROR;
   }
   else if (tiledData & (wallWidth == 0 || wallHeight == 0))
   {
      awarning = WARN_02;
      success = DEF_ERROR;
   }
   else
   {
      success = OK;
   }

   //
   //show warnings
   //
   switch (awarning)
   {
      case WARN_01:
      {
         QString message = tr("The screen size (width, height) is not set.\n\n"
               "Please set the screen size properly.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_01;
         break;
      }
      case WARN_02:
      {
         QString message = tr("The dimension of the whole wall "
               "(width, height) is not set.\n\n"
               "Please set the dimension of the whole wall properly.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_02;
         break;
      }
      default:
      {
         qDebug() << "No warning appeared or warningCode can't be evaluated";
         break;
      }
   }


   //sobald keine Fehler in der Eingabe sind:
   //- wird die projection-Liste an VRCWHost und VRCWHostProjection uebergeben
   //- wird res, size, vPos, floor, typeProj, wallSize, overlap und frame
   //  an VRCWFinal uebergeben
   //
   if (success == OK)
   {
      int trackDimIndex = vrcwList.size() - 3;
      int finalIndex = vrcwList.size() - 2;

      VRCWHost* host = dynamic_cast<VRCWHost*> (vrcwList[index + 1]);
      VRCWHostProjection* hostProjection =
            dynamic_cast<VRCWHostProjection*> (vrcwList[index + 2]);
      VRCWTrackingDim* trackingDim =
            dynamic_cast<VRCWTrackingDim*> (vrcwList[trackDimIndex]);
      VRCWFinal* final = dynamic_cast<VRCWFinal*> (vrcwList[finalIndex]);

      QStringList projection = createProjection();
      host->setCalcNumHosts(projection);
      hostProjection->setGuiProjectionData(projection);

      QVector<int> vPos = projectionVposFloor->getGuiVPos();
      int floor = projectionVposFloor->getGuiFloor();

      trackingDim->setCHeadOffset(vPos);

      //pWallDim is set in createProjecton()
      final->setProjectDimPowerwall(vPos, floor, pWallDim);
   }


   return success;
}

//Entgegennehmen der Parameter von ProjectionHw fuer die Bestimmung
//der Projection
//
void VRCWProjectionDimPowerwall::setProjectionHwParams(const proKind& kind,
      const bool& tiled, const typePro& typeP, const stType& stereo,
      const bool& bothEyes, const int& graka)
{
   kindData = kind;
   tiledData = tiled;
   typePData = typeP;
   stereoData = stereo;
   bothEyesData = bothEyes;
   grakaData = graka;

   QString typePDataStr = typeProToStr(typePData);
   QString kindDataStr = proKindToStr(kindData);

   if (tiledData)
   {
      ui.dimTiledWidget->show();
      ui.dimWidget->hide();
      projectionResSizeTiled->setKindProjection(Powerwall);
      projectionResSizeTiled->setTypeProjection(typePData);

      //set default AspectRatio
      //on first call of setProjectionHwParams()
      //are typePOld and kindOld empty QStrings
      //and we can set the default on startup
      if (typePOld != typePDataStr || kindOld != kindDataStr)
      {
         switch(typePData)
         {
            case Monitor:
            {
               projectionResSizeTiled->setAspRat(_169);
               break;
            }
            case Projector:
            {
               projectionResSizeTiled->setAspRat(_43);
               break;
            }
         }
      }
   }
   else
   {
      ui.dimTiledWidget->hide();
      ui.dimWidget->show();

      //set default AspectRatio
      //on first call of setProjectionHwParams()
      //are typePOld and kindOld empty QStrings
      //and we can set the default on startup
      if (typePOld != typePDataStr || kindOld != kindDataStr)
      {
         switch (typePData)
         {
            case Projector:
            {
               projectionResSize->setGuiAspRat(_43);
               break;
            }
            case Monitor:
            {
               projectionResSize->setGuiAspRat(_169);
               break;
            }
         }
      }
   }

   typePOld = typePDataStr;
   kindOld = kindDataStr;
}


/*****
 * private functions
 *****/

//Auslesen des GUI
//
QVector<int> VRCWProjectionDimPowerwall::getGuiRes() const
{
   if (tiledData)
   {
      return projectionResSizeTiled->getGuiRes();
   }
   else
   {
      return projectionResSize->getGuiRes();
   }
}

QVector<int> VRCWProjectionDimPowerwall::getGuiScreenSize() const
{
   if (tiledData)
   {
      return projectionResSizeTiled->getGuiScreenSize();
   }
   else
   {
      return projectionResSize->getGuiScreenSize();
   }
}

QList<wallVal*> VRCWProjectionDimPowerwall::getGuiPWallDim() const
{
   wallVal* pwv = new wallVal();

   pwv->wall = Front;
   pwv->res = getGuiRes();
   pwv->screenSize = getGuiScreenSize();
   pwv->rowCol = projectionResSizeTiled->getGuiRowCol();
   pwv->typeProj = projectionResSizeTiled->getGuiTypeProj();
   pwv->wallSize = projectionResSizeTiled->getGuiWallSize();
   pwv->overlap = projectionResSizeTiled->getGuiOverlap();
   pwv->frame = projectionResSizeTiled->getGuiFrame();

   QList<wallVal*> listPWV;
   listPWV.prepend(pwv);

   return listPWV;
}

//Erzeugen der Liste der Projektionen
//
QStringList VRCWProjectionDimPowerwall::createProjection()
{
   //get PowerwallDimensions
   pWallDim = getGuiPWallDim();

   //Projection Modus identifizieren
   QString mStr;

   switch (kindData)
   {
      case Powerwall:
      {
         mStr = "P";
         break;
      }
      case _3D_TV:
      {
         mStr = "TV";
         break;
      }
      default:
      {
         //do nothing
         break;
      }
   }

   //tiled
   if (tiledData)
   {
      mStr.append("t");
   }
   else
   {
      mStr.append("f");
   }

   //stereo
   switch (stereoData)
   {
      case passive:
      {
         mStr.append("p");
         break;
      }
      case active:     //active, topBottom, sideBySide, checkerboard, vInterlaced,
      case topBottom:  //cInterleave, hInterlaced and rInterleave are the same
      case sideBySide:
      case checkerboard:
      case vInterlaced:
      case cInterleave:
      case hInterlaced:
      case rInterleave:
      {
         mStr.append("a");
         break;
      }
   }

   //bothEyes (bei active/sideBySide/checkerboard automatisch
   //auf true fuer den Projection Modus)
   if (stereoData == passive && bothEyesData == false)
   {
      mStr.append("f");
   }
   else
   {
      mStr.append("t");
   }


   //Moegliche Modes:
   //Pfpf:  Powerwall - not tiled - passive - one Eye per Graka
   //Pfpt:  Powerwall - not tiled - passive - both Eyes per Graka
   //Pfat:  Powerwall - not tiled - active  - both Eyes per Graka
   //Ptpf:  Powerwall - tiled     - passive - one Eye per Graka
   //Ptpt:  Powerwall - tiled     - passive - both Eyes per Graka
   //Ptat:  Powerwall - tiled     - active  - both Eyes per Graka
   //TVfat: 3D_TV     - not tiled - active  - both Eyes per Graka

   QHash<QString, pMode> strToPMode;
   strToPMode.insert("Pfpf", Pfpf);
   strToPMode.insert("Pfpt", Pfpt);
   strToPMode.insert("Pfat", Pfat);
   strToPMode.insert("Ptpf", Ptpf);
   strToPMode.insert("Ptpt", Ptpt);
   strToPMode.insert("Ptat", Ptat);
   strToPMode.insert("TVfat", TVfat);


   //Projection Modus erzeugen
   QStringList projection;

   switch (strToPMode.value(mStr))
   {
      case Pfpf:
      {
         qDebug() << "Modus 1";

         projection.append("Powerwall - Left Eye");
         projection.append("Powerwall - Right Eye");
         break;
      }
      case Pfpt://Pfpt is handled the same way as Pfat
      case Pfat:
      {
         qDebug() << "Modus 2";

         projection.append("Powerwall - Both Eyes");
         break;
      }
      case Ptpf:
      {
         qDebug() << "Modus 3";

         for (int i = 1; i <= pWallDim[0]->rowCol[0]; ++i )//Row
         {
            QString s;
            QString row = "Row-" % s.setNum(i);

            for (int j = 1; j <= pWallDim[0]->rowCol[1]; ++j)//Column
            {
               QString col = "Col-" % s.setNum(j);
               QString rc = row % "_" % col % " - ";
               projection.append(rc % "Left Eye");
               projection.append(rc % "Right Eye");
            }
         }
         break;
      }
      case Ptpt://Ptpt is handled the same way as Ptat
      case Ptat:
      {
         qDebug() << "Modus 4";

         for (int i = 1; i <= pWallDim[0]->rowCol[0]; ++i )//Row
         {
            QString s;
            QString row = "Row-" % s.setNum(i);

            for (int j = 1; j <= pWallDim[0]->rowCol[1]; ++j)//Column
            {
               QString col = "Col-" % s.setNum(j);
               QString rc = row % "_" % col % " - Both Eyes";
               projection.append(rc);
            }
         }
         break;
      }
      case TVfat:
      {
         qDebug() << "Modus 2TV";

         projection.append("3D-TV - Both Eyes");
         break;

      }
      default:
      {
         qDebug() << "Projection Modus konnte nicht identifiziert werden!!!";
         break;
      }
   }

   return projection;
}
