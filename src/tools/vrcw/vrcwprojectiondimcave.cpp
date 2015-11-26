#include "vrcwprojectiondimcave.h"

#include <QMessageBox>
#include <QStringBuilder>
#include <QDebug>

#include "vrcwhost.h"
#include "vrcwhostprojection.h"
#include "vrcwtrackingdim.h"
#include "vrcwfinal.h"
#include "vrcwprojectionressize.h"
#include "vrcwprojectionressizetiled.h"
#include "vrcwprojectionvposfloor.h"
#include "datatypes.h"


/*****
 * constructor - destructor
 *****/

VRCWProjectionDimCave::VRCWProjectionDimCave(QWidget *parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //GUI erstellen
   //
   //ContainerTab entfernen
   ui.wallConfigTabWidget->removeTab(2);

   //General
   generalDim = new VRCWProjectionResSize(ui.generalDimWidget);
   ui.generalDimWidgetVLayout->addWidget(generalDim);
   generalDim->showCaveConfig();
   generalPos = new VRCWProjectionVposFloor(ui.generalPosWidget);
   ui.generalPosWidgetVLayout->addWidget(generalPos);
   generalPos->showCaveHint();

   //Front
   front = new VRCWProjectionResSize(ui.frontWidget);
   ui.frontWidgetVLayout->addWidget(front);
   frontTiled = new VRCWProjectionResSizeTiled(ui.frontWidget);
   ui.frontWidgetVLayout->addWidget(frontTiled);

   //Left
   left = new VRCWProjectionResSize(ui.leftWidget);
   ui.leftWidgetVLayout->addWidget(left);
   leftTiled = new VRCWProjectionResSizeTiled(ui.leftWidget);
   ui.leftWidgetVLayout->addWidget(leftTiled);

   //Right
   right = new VRCWProjectionResSize(ui.rightWidget);
   ui.rightWidgetVLayout->addWidget(right);
   rightTiled = new VRCWProjectionResSizeTiled(ui.rightWidget);
   ui.rightWidgetVLayout->addWidget(rightTiled);

   //Bottom
   bottom = new VRCWProjectionResSize(ui.bottomWidget);
   ui.bottomWidgetVLayout->addWidget(bottom);
   bottomTiled = new VRCWProjectionResSizeTiled(ui.bottomWidget);
   ui.bottomWidgetVLayout->addWidget(bottomTiled);

   //Top
   top = new VRCWProjectionResSize(ui.topWidget);
   ui.topWidgetVLayout->addWidget(top);
   topTiled = new VRCWProjectionResSizeTiled(ui.topWidget);
   ui.topWidgetVLayout->addWidget(topTiled);

   //Back
   back = new VRCWProjectionResSize(ui.backWidget);
   ui.backWidgetVLayout->addWidget(back);
   backTiled = new VRCWProjectionResSizeTiled(ui.backWidget);
   ui.backWidgetVLayout->addWidget(backTiled);

   //immer mit Tab General beginnen
   ui.wallConfigTabWidget->setCurrentIndex(0);

   //Variablen setzen
   kindData = CAVE;
   tiledData = false;
   typePData = Projector;
   stereoData = passive;
   bothEyesData = false;
   grakaData = 0;

   //Signals and Slots
   //
   //Empfang der Signale aus general, um die res und size-Werte
   //auf den Waenden zu setzen
   connect(generalDim, SIGNAL(aspRatValueChanged()), this,
         SLOT(setWallsAspRat_exec()));
   connect(generalDim, SIGNAL(resValueChanged()), this,
         SLOT(setWallsResWH_exec()));
   connect(generalDim, SIGNAL(wallsResSameConfChanged()), this,
         SLOT(wallsResEnable_exec()));
   connect(generalDim, SIGNAL(wallsResSameConfChanged()), this,
         SLOT(setWallsAspRat_exec()));
   connect(generalDim, SIGNAL(wallsResSameConfChanged()), this,
         SLOT(setWallsResWH_exec()));
   connect(generalDim, SIGNAL(caveDimValueChanged()), this,
         SLOT(setWallsSizeWH_exec()));

   //Empfang der Signale aus den Waenden um die size-Werte auf general
   //und den Waenden zu setzen
   //screenSize bei untiled; wallSize bei tiled
   //
   //front width, caveDim[0] (width)
   connect(front, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   connect(frontTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   //front height, caveDim[2] (height)
   connect(front, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   connect(frontTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   //front calcWidthHeight checked
   connect(front, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(frontTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));

   //left width, caveDim[1] (depth)
   connect(left, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   connect(leftTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   //left height, caveDim[2] (height)
   connect(left, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   connect(leftTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   //left calcWidthHeight checked
   connect(left, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(leftTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));

   //right width, caveDim[1] (depth)
   connect(right, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   connect(rightTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   //right height, caveDim[2] (height)
   connect(right, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   connect(rightTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   //right calcWidthHeight checked
   connect(right, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(rightTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));

   //bottom width, caveDim[0] (width)
   connect(bottom, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   connect(bottomTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   //bottom height, caveDim[1] (depth)
   connect(bottom, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   connect(bottomTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   //bottom calcWidthHeight checked
   connect(bottom, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(bottomTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));

   //top width, caveDim[0] (width)
   connect(top, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   connect(topTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   //top height, caveDim[1] (depth)
   connect(top, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   connect(topTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimDepthWall_exec(int)));
   //top calcWidthHeight checked
   connect(top, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(topTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));

   //back width, caveDim[0] (width)
   connect(back, SIGNAL(sizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   connect(backTiled, SIGNAL(wallSizeWidthValueChanged(int)), this,
         SLOT(setCaveDimWidthWall_exec(int)));
   //back height, caveDim[2] (height)
   connect(back, SIGNAL(sizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   connect(backTiled, SIGNAL(wallSizeHeightValueChanged(int)), this,
         SLOT(setCaveDimHeightWall_exec(int)));
   //back calcWidthHeight checked
   connect(back, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
   connect(backTiled, SIGNAL(calcWidthHeightChecked()), this,
         SLOT(uncheckCalcWidthHeightCheckbox()));
}

VRCWProjectionDimCave::~VRCWProjectionDimCave()
{

}


/*****
 * public functions
 *****/

//Angepasste Version der virtuellen Funktion der Basisklasse VRCWBase
//Bearbeitung und Ueberpruefung der Eingaben im GUI
//
int VRCWProjectionDimCave::processGuiInput(const int& index,
      const QList<VRCWBase*>& vrcwList)
{
   const int WARN_01 = 1;
   const int ERROR_01 = 301;
   const int OK = 3;//same as in VRCWProjectionDimPowerwall

   int success = DEF_ERROR;
   int awarning = DEF_WARN;

   //
   //doing some checks
   //

   //set variables
   int caveWidth = caveDim[0];
   int caveDepth = caveDim[1];
   int caveHeight = caveDim[2];

   if (caveWidth == 0 || caveDepth == 0 || caveHeight == 0)
   {
      awarning = WARN_01;
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
         QString message = tr("Some of the CAVE dimensions "
               "(width, depth, height) are not set.\n\n"
               "Please set the CAVE dimensions and check on every "
               "configured wall that the specified screen size or "
               "the dimension of the whole wall are set properly.");
         QMessageBox::warning(this, tr("Configuration"), message,
               QMessageBox::Ok);

         success = ERROR_01;
         break;
      }
      case DEF_WARN:
      {
         //do nothing
         break;
      }
      default:
      {
          qDebug() << "WarningCode can't be evaluated";
          break;
      }
   }


   //sobald keine Fehler in der Eingabe sind:
   //- wird die projection-Liste an VRCWHost und VRCWHostProjection uebergeben
   //- wird vPos, floor, caveDim und caveWallDim an VRCWFinal uebergeben
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

      //get all values
      //
      QStringList projection = createProjection();
      host->setCalcNumHosts(projection);
      hostProjection->setGuiProjectionData(projection);

      zPoint zeroP = getGuiZeroPoint();
      QVector<int> vPos = generalPos->getGuiVPos();
      int floor = generalPos->getGuiFloor();
      caveDim = getGuiCaveDim();

      trackingDim->setCHeadOffset(vPos);

      //caveWallDim is set in createProjecton()
      final->setProjectDimCave(zeroP, vPos, floor, caveDim, caveWallDim);
   }

   return success;
}

//Setzen der Variablen wallsToConfigure
//und anzeigen der zu konfigurierenden Waende
//
void VRCWProjectionDimCave::setGuiWallsToConfigure(const QList<cWall>& walls)
{
   wallsToConfigure = walls;

   int tabCount = ui.wallConfigTabWidget->count();

   //alle Tabs bis auf Index=0=General loeschen
   //(diese wird immer angezeigt und ist immer an der gleichen Stelle)
   for (int i = (tabCount - 1); i > 0; --i)
   {
      ui.wallConfigTabWidget->removeTab(i);
   }

   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   //get value of zero point and disable zero point for bottom by default
   zPoint zeroP = getGuiZeroPoint();
   ui.bottomWallRadioButton->setDisabled(true);

   //Tabs fuer zu konfigurierende Walls erzeugen
   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            ui.wallConfigTabWidget->addTab(ui.frontPage, "Front");
            utWall = front;
            tWall = frontTiled;
            break;
         }
         case Left:
         {
            ui.wallConfigTabWidget->addTab(ui.leftPage, "Left");
            utWall = left;
            tWall = leftTiled;
            break;
         }
         case Right:
         {
            ui.wallConfigTabWidget->addTab(ui.rightPage, "Right");
            utWall = right;
            tWall = rightTiled;
            break;
         }
         case Bottom:
         {
            ui.wallConfigTabWidget->addTab(ui.bottomPage, "Bottom");
            utWall = bottom;
            tWall = bottomTiled;

            //enable zero point radioButton
            ui.bottomWallRadioButton->setEnabled(true);
            break;
         }
         case Top:
         {
            ui.wallConfigTabWidget->addTab(ui.topPage, "Top");
            utWall = top;
            tWall = topTiled;
            break;
         }
         case Back:
         {
            ui.wallConfigTabWidget->addTab(ui.backPage, "Back");
            utWall = back;
            tWall = backTiled;
            break;
         }
      }

      utWall->checkCalcWidthHeight(false);
      tWall->checkCalcWidthHeight(false);
   }

   //zum Uebertragen der resolution general Werte auf die Tabs
   //und deaktivieren der Resolution spinboxen auf den Tabs
   setWallsResWH_exec();
   wallsResEnable_exec();

   //set the caveDim variable and the Screensize values on the tabs
   caveDim = getGuiCaveDim();
   setWallsSizeWH();

   //set zero point
   setZeroPoint(zeroP);

   if (zeroP == boWall && !ui.bottomWallRadioButton->isEnabled())
   {
      ui.caveDimRadioButton->setChecked(true);
   }

   zeroPointChanged();
}

//Entgegennehmen der Parameter von ProjectionHw fuer die Bestimmung
//der Projection
//
void VRCWProjectionDimCave::setProjectionHwParams(const proKind& kind,
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

   //show/hide tiledDims
   if (tiledData)
   {
      showTiledDim();
   }
   else
   {
      showTiledDim(false);
   }

   //set default AspectRatio
   //on first call of setProjectionHwParams() is typePOld an empty QString
   //and we can set the default on startup
   if (typePOld != typePDataStr)
   {
      switch (typePData)
      {
         case Projector:
         {
            generalDim->setGuiAspRat(_43);
            break;
         }
         case Monitor:
         {
            generalDim->setGuiAspRat(_169);
            break;
         }
      }
   }

   typePOld = typePDataStr;
}


/*****
 * private functions
 *****/

//set the aspect ratio
//from general in selected tabs depending on wallSameResConf checkBox
//
void VRCWProjectionDimCave::setWallsAspRat() const
{
   aspRat aR = getGuiAspectRatioGeneral();
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            utWall = front;
            tWall = frontTiled;
            break;
         }
         case Left:
         {
            utWall = left;
            tWall = leftTiled;
            break;
         }
         case Right:
         {
            utWall = right;
            tWall = rightTiled;
            break;
         }
         case Bottom:
         {
            utWall = bottom;
            tWall = bottomTiled;
            break;
         }
         case Top:
         {
            utWall = top;
            tWall = topTiled;
            break;
         }
         case Back:
         {
            utWall = back;
            tWall = backTiled;
            break;
         }
      }
      utWall->setGuiAspRat(aR);
      tWall->setGuiAspRat(aR);
   }
}

//set the predefined or user defined resolution
//from general in selected tabs depending on wallSameResConf checkBox
//
void VRCWProjectionDimCave::setWallsResWH() const
{
   QVector<int> resWH = getGuiResGeneral();
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            utWall = front;
            tWall = frontTiled;
            break;
         }
         case Left:
         {
            utWall = left;
            tWall = leftTiled;
            break;
         }
         case Right:
         {
            utWall = right;
            tWall = rightTiled;
            break;
         }
         case Bottom:
         {
            utWall = bottom;
            tWall = bottomTiled;
            break;
         }
         case Top:
         {
            utWall = top;
            tWall = topTiled;
            break;
         }
         case Back:
         {
            utWall = back;
            tWall = backTiled;
            break;
         }
      }
      utWall->setGuiRes(resWH);
      tWall->setGuiRes(resWH);
   }
}

//wallSameResConfCheckBox ueberpruefen und auf vorhandenen Tabs
//die Resolution width und height aktivieren/deaktivieren
//
void VRCWProjectionDimCave::wallsResEnable(const bool& yes) const
{
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            utWall = front;
            tWall = frontTiled;
            break;
         }
         case Left:
         {
            utWall = left;
            tWall = leftTiled;
            break;
         }
         case Right:
         {
            utWall = right;
            tWall = rightTiled;
            break;
         }
         case Bottom:
         {
            utWall = bottom;
            tWall = bottomTiled;
            break;
         }
         case Top:
         {
            utWall = top;
            tWall = topTiled;
            break;
         }
         case Back:
         {
            utWall = back;
            tWall = backTiled;
            break;
         }
      }
      utWall->enableRes(yes);
      tWall->enableRes(yes);
   }
}

//setzen der ScreenSize-Werte auf den Tabs
//
void VRCWProjectionDimCave::setWallsSizeWH() const
{
   //width = wallSize[0]; height = wallSize[1]
   QVector<int> wallSize(2,0);
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            //width and height are connected with other walls
            utWall = front;
            tWall = frontTiled;
            wallSize[0] = caveDim[0];
            wallSize[1] = caveDim[2];
            break;
         }
         case Left:
         {
            //width can be different than other walls (not connected)
            //height is connected with other walls
            utWall = left;
            tWall = leftTiled;

            if (tiledData)
            {
               wallSize = tWall->getGuiScreenSize();
            }
            else
            {
               wallSize = utWall->getGuiScreenSize();
            }

            //at start the General tab is visible first, so we can set
            //the cave dimensions here and with this the values of the walls
            //if the wall values change, they became decisive
            if (wallSize[0] == 0)
            {
               wallSize[0] = caveDim[1];
            }

            //wallSize[0] = width is taken from left wall screen size
            wallSize[1] = caveDim[2];
            break;
         }
         case Right:
         {
            //width can be different than other walls (not connected)
            //height is connected with other walls
            utWall = right;
            tWall = rightTiled;

            if (tiledData)
            {
               wallSize = tWall->getGuiScreenSize();
            }
            else
            {
               wallSize = utWall->getGuiScreenSize();
            }

            //at start the General tab is visible first, so we can set
            //the cave dimensions here and with this the values of the walls
            //if the wall values change, they became decisive
            if (wallSize[0] == 0)
            {
               wallSize[0] = caveDim[1];
            }

            //wallSize[0] = width is taken from right wall screen size
            wallSize[1] = caveDim[2];
            break;
         }
         case Bottom:
         {
            //width is connected with other walls
            //height can be different than other walls (not connected)
            utWall = bottom;
            tWall = bottomTiled;

            if (tiledData)
            {
               wallSize = tWall->getGuiScreenSize();
            }
            else
            {
               wallSize = utWall->getGuiScreenSize();
            }

            //at start the General tab is visible first, so we can set
            //the cave dimensions here and with this the values of the walls
            //if the wall values change, they became decisive
            if (wallSize[1] == 0)
            {
               wallSize[1] = caveDim[1];
            }

            wallSize[0] = caveDim[0];
            //wallSize[1] = height is taken from bottom wall screen size
            break;
         }
         case Top:
         {
            //width is connected with other walls
            //height can be different than other walls (not connected)
            utWall = top;
            tWall = topTiled;


            if (tiledData)
            {
               wallSize = tWall->getGuiScreenSize();
            }
            else
            {
               wallSize = utWall->getGuiScreenSize();
            }

            //at start the General tab is visible first, so we can set
            //the cave dimensions here and with this the values of the walls
            //if the wall values change, they became decisive
            if (wallSize[1] == 0)
            {
               wallSize[1] = caveDim[1];
            }

            wallSize[0] = caveDim[0];
            //wallSize[1] = height is taken from top wall screen size
            break;
         }
         case Back:
         {
            //width and height are connected with other walls
            utWall = back;
            tWall = backTiled;
            wallSize[0] = caveDim[0];
            wallSize[1] = caveDim[2];
            break;
         }
      }
      utWall->checkCalcWidthHeight(false);
      utWall->setGuiScreenSize(wallSize);
      tWall->setGuiWallSize(wallSize);
   }

   //cave dimension can only be set on general tab at the start and as long
   //the value in the spinbox for width, depth and height are zero
   generalDim->setGuiCaveDimSbDisabled();
}

//setzen des zero point
//
void VRCWProjectionDimCave::setZeroPoint(const zPoint& zp)
{
   switch (zp)
   {
      case fWall:
      {
         ui.frontWallRadioButton->setChecked(true);
         break;
      }
      case boWall:
      {
         ui.bottomWallRadioButton->setChecked(true);
         break;
      }
      case cDim:
      {
         ui.caveDimRadioButton->setChecked(true);
         break;
      }
   }
}


//show/hide tiledDims or untiledDims
//
void VRCWProjectionDimCave::showTiledDim(const bool& yes) const
{
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   if (yes)
   {
      for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
      {
         switch (wallsToConfigure[i])
         {
            case Front:
            {
               utWall = front;
               tWall = frontTiled;
               break;
            }
            case Left:
            {
               utWall = left;
               tWall = leftTiled;
               break;
            }
            case Right:
            {
               utWall = right;
               tWall = rightTiled;
               break;
            }
            case Bottom:
            {
               utWall = bottom;
               tWall = bottomTiled;
               break;
            }
            case Top:
            {
               utWall = top;
               tWall = topTiled;
               break;
            }
            case Back:
            {
               utWall = back;
               tWall = backTiled;
               break;
            }
         }
         utWall->hide();
         tWall->show();
         tWall->setTypeProjection(typePData);

         if (getGuiWallsSameResConf())
         {
            aspRat  aR;
            switch(typePData)
            {
               case Monitor:
               {
                  aR = _169;
                  break;
               }
               case Projector:
               {
                  aR = _43;
                  break;
               }
            }

            tWall->setAspRat(aR);
         }
      }
   }
   else
   {
      for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
      {
         switch (wallsToConfigure[i])
         {
            case Front:
            {
               utWall = front;
               tWall = frontTiled;
               break;
            }
            case Left:
            {
               utWall = left;
               tWall = leftTiled;
               break;
            }
            case Right:
            {
               utWall = right;
               tWall = rightTiled;
               break;
            }
            case Bottom:
            {
               utWall = bottom;
               tWall = bottomTiled;
               break;
            }
            case Top:
            {
               utWall = top;
               tWall = topTiled;
               break;
            }
            case Back:
            {
               utWall = back;
               tWall = backTiled;
               break;
            }
         }
         utWall->show();
         tWall->hide();
      }
   }
}


// Auslesen des GUI
//
//General
QVector<int> VRCWProjectionDimCave::getGuiResGeneral() const
{
   return generalDim->getGuiRes();
}

aspRat VRCWProjectionDimCave::getGuiAspectRatioGeneral() const
{
   return generalDim->getGuiAspectRatio();
}

QVector<int> VRCWProjectionDimCave::getGuiCaveDim() const
{
   return generalDim->getGuiCaveDim();
}

bool VRCWProjectionDimCave::getGuiWallsSameResConf() const
{
   return generalDim->getGuiWallsSameResConf();
}

QList<wallVal*> VRCWProjectionDimCave::getGuiCaveWallDim() const
{
   QList<wallVal*> listCWV;

   if (tiledData)
   {
      //Werte in den Tabs der konfigurierten Walls auslesen
      for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
      {
         //placeholder variable
         VRCWProjectionResSizeTiled* rst;
         //tiled CaveWall Variable
         wallVal* cwvt = new wallVal();

         switch (wallsToConfigure[i])
         {
            case Front:
            {
               rst = frontTiled;
               break;
            }
            case Left:
            {
               rst = leftTiled;
               break;
            }
            case Right:
            {
               rst = rightTiled;
               break;
            }
            case Bottom:
            {
               rst = bottomTiled;
               break;
            }
            case Top:
            {
               rst = topTiled;
               break;
            }
            case Back:
            {
               rst = backTiled;
               break;
            }
         }
         cwvt->wall = wallsToConfigure[i];
         cwvt->res = rst->getGuiRes();
         cwvt->screenSize = rst->getGuiScreenSize();
         cwvt->rowCol = rst->getGuiRowCol();
         cwvt->typeProj = rst->getGuiTypeProj();
         cwvt->wallSize = rst->getGuiWallSize();
         cwvt->overlap = rst->getGuiOverlap();
         cwvt->frame = rst->getGuiFrame();

         listCWV.append(cwvt);
      }
   }
   else
   {
      //Werte in den Tabs der konfigurierten Walls auslesen
      for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
      {
         //placeholder variable
         VRCWProjectionResSize* rs;
         //untiled CaveWall Variable
         wallVal* cwv = new wallVal();

         switch (wallsToConfigure[i])
         {
            case Front:
            {
               rs = front;
               break;
            }
            case Left:
            {
               rs = left;
               break;
            }
            case Right:
            {
               rs = right;
               break;
            }
            case Bottom:
            {
               rs = bottom;
               break;
            }
            case Top:
            {
               rs = top;
              break;
            }
            case Back:
            {
               rs = back;
               break;
            }
         }
         cwv->wall = wallsToConfigure[i];
         cwv->res = rs->getGuiRes();
         cwv->screenSize = rs->getGuiScreenSize();

         listCWV.append(cwv);
      }
   }

   //the returning list shouldn't be empty
   if (listCWV.size() <= 0)
   {
      wallVal* cwv = new wallVal();
      listCWV.append(cwv);
   }

   return listCWV;
}

zPoint VRCWProjectionDimCave::getGuiZeroPoint() const
{
   zPoint zp;

   if (ui.frontWallRadioButton->isChecked())
   {
      zp = fWall;
   }
   else if (ui.bottomWallRadioButton->isChecked())
   {
      zp = boWall;
   }
   else
   {
      zp = cDim;
   }

   return zp;
}

//Erzeugen der Liste der Projektionen
//
QStringList VRCWProjectionDimCave::createProjection()
{
   //get caveWallDimensions
   caveWallDim = getGuiCaveWallDim();

   //Projection Modus identifizieren
   //kind == "CAVE"
   QString mStr = "C";

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
      case active:
      case topBottom:    //active, topBottom, sideBySide, checkerboard, vInterlaced,
      case sideBySide:   //cInterleave, hInterlaced and rInterleave are the same
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
   //Cfpf: CAVE - not tiled - passive - one Eye per Graka
   //Cfpt: CAVE - not tiled - passive - both Eyes per Graka
   //Cfat: CAVE - not tiled - active  - both Eyes per Graka
   //Ctpf: CAVE - tiled     - passive - one Eye per Graka
   //Ctpt: CAVE - tiled     - passive - both Eyes per Graka
   //Ctat: CAVE - tiled     - active  - both Eyes per Graka

   QHash<QString, cMode> strToCMode;
   strToCMode.insert("Cfpf", Cfpf);
   strToCMode.insert("Cfpt", Cfpt);
   strToCMode.insert("Cfat", Cfat);
   strToCMode.insert("Ctpf", Ctpf);
   strToCMode.insert("Ctpt", Ctpt);
   strToCMode.insert("Ctat", Ctat);


   //Projection Modus erzeugen
   QStringList projection;

   switch (strToCMode.value(mStr))
   {
      case Cfpf:
      {
         qDebug() << "Modus 5";

         for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
         {
            QString wall = cWallToStr(wallsToConfigure[i]) % "-Wall - ";
            projection.append(wall % "Left Eye");
            projection.append(wall % "Right Eye");
         }
         break;
      }
      case Cfpt://Cfpt is handled the same way as Cfat
      case Cfat:
      {
         qDebug() << "Modus 6";

         for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
         {
            QString wall = cWallToStr(wallsToConfigure[i]) %
                  "-Wall - Both Eyes";
            projection.append(wall);
         }
         break;
      }
      case Ctpf:
      {
         qDebug() << "Modus 7";

         for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
         {
            QString wall = cWallToStr(wallsToConfigure[i]);

            //rowCol fuer wall bestimmen
            int j = 0;
            while (wallsToConfigure[i] != caveWallDim[j]->wall)
            {
               ++j;
            }
            QVector<int> rowColT = caveWallDim[j]->rowCol;

            for (int k = 1; k <= rowColT[0]; ++k )
            {
               QString s;
               QString row = "Row-" % s.setNum(k);

               for (int l = 1; l <= rowColT[1]; ++l)
               {
                  QString col = "Col-" % s.setNum(l);
                  QString wrc = wall % "_" % row % "_" % col % " - ";
                  projection.append(wrc % "Left Eye");
                  projection.append(wrc % "Right Eye");
               }
            }
         }
         break;
      }
      case Ctpt://Ctpt is handled the same way as Ctat
      case Ctat:
      {
         qDebug() << "Modus 8";

         for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
         {
            QString wall = cWallToStr(wallsToConfigure[i]);

            //rowCol fuer wall bestimmen
            int j = 0;
            while (wallsToConfigure[i] != caveWallDim[j]->wall)
            {
               ++j;
            }
            QVector<int> rowColT = caveWallDim[j]->rowCol;

            for (int k = 1; k <= rowColT[0]; ++k )
            {
               QString s;
               QString row = "Row-" % s.setNum(k);

               for (int l = 1; l <= rowColT[1]; ++l)
               {
                  QString col = "Col-" % s.setNum(l);
                  QString wrc = wall % "_" % row % "_" % col % " - ";
                  projection.append(wrc % "Both Eyes");
               }
            }
         }
         break;
      }
      default:
      {
         qDebug() << "Projection Mode can't be identified!!!";
         break;
      }
   }

   return projection;
}


/*****
 * private slots
 *****/

//set the aspect ratio
//from general in selected tabs depending on wallSameResConf checkBox
//
void VRCWProjectionDimCave::setWallsAspRat_exec() const
{
   if (getGuiWallsSameResConf())
   {
      setWallsAspRat();
   }
}

//set the predefined oder user defined resolution
//from general in selected tabs depending on wallSameResConf checkBox
//
void VRCWProjectionDimCave::setWallsResWH_exec() const
{
   if (getGuiWallsSameResConf())
   {
      setWallsResWH();
   }
}

//enable or disable the predefined resolution and/or user defined width/height
//spinbox depending on wallSameResConfCheckBox
//
void VRCWProjectionDimCave::wallsResEnable_exec() const
{
   if (getGuiWallsSameResConf())
   {
      wallsResEnable(false);
   }
   else
   {
      wallsResEnable(true);
   }
}

//setzen der Werte aus caveDim general in Size in den vorhandenen Tabs
//
void VRCWProjectionDimCave::setWallsSizeWH_exec()
{
   QVector<int> newCaveDim = getGuiCaveDim();

   if (caveDim != newCaveDim)
   {
      caveDim = newCaveDim;
      setWallsSizeWH();
   }

   //cave dimension can only be set on general tab at the start and as long
   //the value in the spinbox for width, depth and height are zero
   generalDim->setGuiCaveDimSbDisabled();
}

//uncheck the calcWidthHeightCheckbox on the tabs
//
void VRCWProjectionDimCave::uncheckCalcWidthHeightCheckbox() const
{
   //placeholder variables
   VRCWProjectionResSize* utWall;
   VRCWProjectionResSizeTiled* tWall;

   for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
   {
      switch (wallsToConfigure[i])
      {
         case Front:
         {
            utWall = front;
            tWall = frontTiled;
            break;
         }
         case Left:
         {
            utWall = left;
            tWall = leftTiled;
            break;
         }
         case Right:
         {
            utWall = right;
            tWall = rightTiled;
            break;
         }
         case Bottom:
         {
            utWall = bottom;
            tWall = bottomTiled;
            break;
         }
         case Top:
         {
            utWall = top;
            tWall = topTiled;
            break;
         }
         case Back:
         {
            utWall = back;
            tWall = backTiled;
            break;
         }
      }
      utWall->checkCalcWidthHeight(false);
      tWall->checkCalcWidthHeight(false);
   }
}

//Entgegennehmen des size Wertes fuer CaveDim width und setzen auf den Tabs
//
void VRCWProjectionDimCave::setCaveDimWidthWall_exec(const int& newValue)
{
   if (caveDim[0] != newValue)
   {
      caveDim[0] = newValue;
      //set CaveDim in GUI
      generalDim->setGuiCaveDim(caveDim);
      //set wall size in GUI
      setWallsSizeWH();
   }
}

//Entgegennehmen des size Wertes fuer CaveDim depth und setzen auf den Tabs
//
void VRCWProjectionDimCave::setCaveDimDepthWall_exec(const int& newValue)
{
   if (caveDim[1] != newValue)
   {
      //set CaveDim in GUI
      //the cave depth depends on left, right, top, bottom
      //all of them can have different widths/heights (is the cave depth)
      //for cave depth the largest value of them is decisive
      int maxDepth = 0;

      //placeholder variables
      VRCWProjectionResSize* utWall;
      VRCWProjectionResSizeTiled* tWall;

      for (QList<cWall>::size_type i = 0; i < wallsToConfigure.size(); ++i)
      {
         //find the specific wall
         switch (wallsToConfigure[i])
         {
            case Left:
            {
               utWall = left;
               tWall = leftTiled;
               break;
            }
            case Right:
            {
               utWall = right;
               tWall = rightTiled;
               break;
            }
            case Bottom:
            {
               utWall = bottom;
               tWall = bottomTiled;
               break;
            }
            case Top:
            {
               utWall = top;
               tWall = topTiled;
               break;
            }
            case Front:
            case Back:
            {
               //do nothing
               break;
            }
         }

         //Screen size of wall and with this the cave depth
         QVector<int> partScreenSize;
         int partDepth = 0;

         switch (wallsToConfigure[i])
         {
            case Left:
            case Right:
            {
               if (tiledData)
               {
                  partScreenSize = tWall->getGuiWallSize();
               }
               else
               {
                  partScreenSize = utWall->getGuiScreenSize();
               }

               //wall width == cave depth
               partDepth = partScreenSize[0];
               break;
            }
            case Bottom:
            case Top:
            {
               if (tiledData)
               {
                  partScreenSize = tWall->getGuiWallSize();
               }
               else
               {
                  partScreenSize = utWall->getGuiScreenSize();
               }

               //wall height == cave depth
               partDepth = partScreenSize[1];
               break;
            }
            case Front:
            case Back:
            {
               //do nothing
               break;
            }
         }

         if (partDepth > maxDepth)
         {
            maxDepth = partDepth;
         }
      }

      caveDim[1] = maxDepth;
      //set CaveDim in GUI
      generalDim->setGuiCaveDim(caveDim);

      //cave dimension can only be set on general tab at the start and as long
      //the value in the spinbox for width, depth and height are zero
      generalDim->setGuiCaveDimSbDisabled();
   }
}

//Entgegennehmen des size Wertes fuer CaveDim height und setzen auf den Tabs
//
void VRCWProjectionDimCave::setCaveDimHeightWall_exec(const int& newValue)
{
   if (caveDim[2] != newValue)
   {
      caveDim[2] = newValue;
      //set CaveDim in GUI
      generalDim->setGuiCaveDim(caveDim);
      //set wall size in GUI
      setWallsSizeWH();
   }
}

//location of zero point changed and set description
//
void VRCWProjectionDimCave::zeroPointChanged()
{
   zPoint zp = getGuiZeroPoint();

   switch (zp)
   {
      case fWall:
      {
         ui.zeroPointDescLabel->setText("The zero point is located in the "
                  "middle of the Front wall.");
         break;
      }
      case boWall:
      {
         ui.zeroPointDescLabel->setText("The zero point is located in the "
                  "middle of the Bottom wall.");
         break;
      }
      case cDim:
      default:
      {
         ui.zeroPointDescLabel->setText("The zero point is located in the "
               "middle of the volume that the walls (CAVE dimension) "
               "enclose.");
         break;
      }
   }
}

