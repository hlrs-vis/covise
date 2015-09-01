#include "vrcwfinal.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QStringBuilder>
#include <math.h>
#ifdef WIN32
static inline double round(double val)
{
    return floor(val + 0.5);
}
inline double trunc(double x)
{
    return (x>0) ? floor(x) : ceil(x);
}
#endif

#include "datatypes.h"


/*****
 * constructor - destructor
 *****/

VRCWFinal::VRCWFinal(QWidget* parent) :
   QWidget(parent)
{
   ui.setupUi(this);

   //set variables
   coValData = new configVal();
   confModified = false;
   kindData = Powerwall;
   tiledData = false;
   stereoData = passive;
   bothEyesData = false;
   grakaData = 0;
   cMonData = false;
   cMon3DOrientData = leftOf;
   floorPwData = 0;
   zeroPData = cDim;
   floorCData = 0;
   osData = Linux;
   execData = ssh;
   thwvdData = new trackHwValDim();

   //setup Gui
   ui.configTabWidget->setCurrentIndex(0);
   ui.configUserTextEdit->setReadOnly(true);
   ui.configVrcwTextEdit->setReadOnly(true);
}

VRCWFinal::~VRCWFinal()
{

}


/*****
 * public functions
 *****/

//Entgegennehmen des Speicherortes und des Namens fuer
//die zu erzeugende configUser.xml und configVrcw.xml
//
void VRCWFinal::setstart(configVal*& coVal)
{
   coValData = coVal;

   //set and generate filenames
   confNameUser = coValData->coConfName;
   confNameVrcw = confNameUser;
   confNameVrcw.replace(".xml", ".vrcw.xml");

   //Path to storage directory
   QString pathTmp;

   if (coValData->coConfDirWritable)
   {
      pathTmp = coValData->coConfDir;
   }
   else
   {
      pathTmp = coValData->coConfTmpDir;
   }

   confStorPath = QDir::toNativeSeparators(pathTmp + "/");

}


//Entgegennehmen von ProjectionKind, Tiled, Stereo, bothEyes, No of Graka
//Number of Rows/Columns, List of CaveWalls von ProjectionHW
//
void VRCWFinal::setProjectionHw(const proKind& kind, const bool& tiled,
      const stType& stereo, const bool& bothEyes, const int& graka,
      const bool& cMon, const QVector<int>& cMonRes,
      const loRoOrient& cMon3DOrient)
{
   kindData = kind;
   tiledData = tiled;
   stereoData = stereo;
   bothEyesData = bothEyes;
   grakaData = graka;
   cMonData = cMon;
   cMonResData = cMonRes;
   cMon3DOrientData = cMon3DOrient;
}

//Entgegennehmen von viewerPos, floorHeight und Bezeichnung der wall,
//resolution, size, rowCol, typeProj, wallSize, overlap und frame
//von VRCWProjectionDimPowerwall
//
void VRCWFinal::setProjectDimPowerwall(const QVector<int>& vPosPw,
      const int& floorPw, const QList<wallVal*>& pWallDim)
{
   vPosPwData = vPosPw;
   floorPwData = floorPw;
   pWallDimData = pWallDim;
}

//Entgegennehmen von viewerPos, floorHeight, Cave-Abmessungen
//und Bezeichnung der wall, resolution, size, rowCol, typeProj, wallSize,
//overlap und frame fuer die einzelnen Waende von VRCWProjectionDimCave
//
void VRCWFinal::setProjectDimCave(const zPoint& zeroP,
      const QVector<int>& vPosC, const int& floorC,
      const QVector<int>& caveDim, const QList<wallVal*>& caveWallDim)
{
   zeroPData = zeroP;
   vPosCData = vPosC;
   floorCData = floorC;
   //caveDimData holds the width, depth and height of the cave
   //with and height are values of front and back wall
   //the cave depth depends on left, right, top, bottom
   //all of them can have different widths/heights (is the cave depth)
   //for cave depth the largest value of them is decisive
   //front and back are both connected to the largest value
   //left, right, bottom, top are connected to the front wall
   caveDimData = caveDim;
   caveWallDimData = caveWallDim;
}

//Entgegennehmen von OS und executionMode von VRCWHost
//
void VRCWFinal::setHost(const opSys& os, const execMode& exec)
{
   osData = os;
   execData = exec;
}

//Entgegennehmen von coviseGui und hostProjection von VRCWHostProjection
//
void VRCWFinal::setHostProjection(const QString& coviseGuiHost,
      const QList<QStringList>& hostProjection)
{
   coviseGuiHostData = coviseGuiHost;
   hostProjectionData = hostProjection;

   //create list of allHosts and renderHosts
   allHostnames();
}

//Entgegennehmen von tracking system, button system, button device,
//button driver, num of hand/head sensors, body index, num persons,
//tracked persons
//Entgegennehmen von Tracking, Hand und Head Offset und Orientation
//
void VRCWFinal::setTrackingValDim(trackHwValDim*& thwvd)
{
   thwvdData = thwvd;
}


//Generierung der configs
//
void VRCWFinal::createXmlConfigs()
{
   //generate configUser.xml
   createXmlConfigUser();

   //generate configVrcw.xml
   createXmlConfigVrcw();

   //set names in TabWiget: 0 = configUser, 1 = configVrcw
   ui.configTabWidget->setTabText(0, confNameUser);
   ui.configTabWidget->setTabText(1, confNameVrcw);

   //set the description in the header
   ui.description->setText("Check, Edit, Save " + confNameUser);
}

//Generierung der configUser.xml
//
void VRCWFinal::createXmlConfigUser()
{
   configUser.generateConfigUser(confNameVrcw);
   ui.configUserTextEdit->setPlainText(addNlToStrOut(configUser.strOutUser()));
}

//Generierung der configVrcw.xml
//
void VRCWFinal::createXmlConfigVrcw()
{
   //initialize configVrcw with some elements
   configVrcw.initializeDomDocVrcw();

   //generate data for SceneSize, ViewerPos und floorHeight in configVrcw.xml
   genSceneViewerFloor();

   //generate data for Num{Pipes,Windows,Screens} and Stereo in configVrcw.xml
   genNumPipesWindowsScreensStereo();

   //generate data for MultiPC in configVrcw.xml
   genMultiPC();

   //generate data for PipeConfig entries for each host in configVrcw.xml
   genPipeConfig();

   //generate data for WindowConfig entries for each host in configVrcw.xml
   genWindowConfig();

   //generate data for ChannelConfig and ViewportConfig entries
   //for each host in configVrcw.xml
   genChannelViewportConfig();

   //generate data for ScreenConfig entries for each host in configVrcw.xml
   genScreenConfig();

   //generate data for Input entry in configVrcw.xml
   genInput();

   //write the configVrcw into the editor
   ui.configVrcwTextEdit->setPlainText(addNlToStrOut(configVrcw.strOutVrcw()));
}

//Save Config
//
bool VRCWFinal::saveConfig()
{
   return save_exec();
}


/*****
 * private functions
 *****/

//generate data for SceneSize, ViewerPos und floorHeight in configVrcw.xml
//
void VRCWFinal::genSceneViewerFloor()
{
   int sceneSize = 0;
   QVector<int> vPos(2,0);
   int floor = 0;

   //TILED!

   switch (kindData)
   {
      case Powerwall:
      case _3D_TV://_3D_TV is a untiled Powerwall
      {
         if (tiledData)
         {
            sceneSize = pWallDimData[0]->wallSize[1];
         }
         else
         {
            sceneSize = pWallDimData[0]->screenSize[1];
         }

         vPos = vPosPwData;
         floor = floorPwData;
         break;
      }
      case CAVE:
      {
         //sceneSize == height of front wall
         //untiled: caveDimData[2] == screenSize[1]
         //tiled: caveDimData[2] == wallSize[1]
         sceneSize = caveDimData[2];
         vPos = vPosCData;
         floor = floorCData;
         break;
      }
   }

   configVrcw.sceneViewerFloor(sceneSize, vPos, floor);
}

//generate data for Num{Pipes,Windows,Screens} and Stereo in configVrcw.xml
//
void VRCWFinal::genNumPipesWindowsScreensStereo()
{
   int pipes = grakaData;
   int windows = grakaData;
   int screens = grakaData;

   switch (kindData)
   {
      case Powerwall:
      case CAVE:
      {
         //bothEyesData can only be false, if passive is selected
         //otherwise always true
         if (bothEyesData)
         {
            //for passive: Linux Twinview LeftOf or RightOf
            screens = 2 * grakaData;
         }
         break;
      }
      case _3D_TV:
      {
         //there is always 1 graphic card and 2 channels/screens
         screens = 2;
         break;
      }
   }

   configVrcw.numPipesWindowsScreensStereo(pipes, windows, screens);
}

//generate data for MultiPC in configVrcw.xml
//
void VRCWFinal::genMultiPC()
{
   QStringList hosts = allHosts;
   hosts.removeAll(coviseGuiHostData);

   if (!hosts.isEmpty())
   {
      configVrcw.multiPC(osData, execData, coviseGuiHostData, hosts);
   }
}

//generate data for PipeConfig entries for each host in configVrcw.xml
//
void VRCWFinal::genPipeConfig()
{
   for (QStringList::size_type i = 0; i < allHosts.size(); ++i)
   {
      configVrcw.localPipe(allHosts[i]);
   }
}

//generate data for WindowConfig entries for each host in configVrcw.xml
//
void VRCWFinal::genWindowConfig()
{
   QList<wallVal*> wallDim;

   switch (kindData)
   {
      case Powerwall:
      case _3D_TV://_3D_TV is a untiled Powerwall
      {
         wallDim = pWallDimData;
         break;
      }
      case CAVE:
      {
         wallDim = caveWallDimData;
         break;
      }
   }

   //render hosts
   for (QList<QStringList>::size_type i=0; i < hostProjectionData.size(); ++i)
   {
      QString host;
      QString comment;

      host = hostProjectionData[i][2];
      comment = hostProjectionData[i][0] % "_" %
            hostProjectionData[i][1] % "-Eye";

      if (hostProjectionData[i][1] == "Both")
      {
         comment += "s";
      }

      QStringList wallStrings = hostProjectionData[i][0].split(QRegExp("[-_]"),
            QString::SkipEmptyParts);
      cWall wall = strToCWall(wallStrings[0]);

      //determine the fitting index in caveWallDimData or pWallDimData for wall
      //wallDim == pWallDimData: wallDim.size == 1 and Index == 0
      int j = 0;
      while ( (wallDim.size() > 1) && (wall != wallDim[j]->wall) )
      {
         ++j;
      }

      //Resolution[Pixel]
      int width = wallDim[j]->res[0];
      int height = wallDim[j]->res[1];

      //Powerwall/Cave passiv stereo:
      //wenn eine GraKa beide Augen anzeigen soll, benutzen wir
      //TwinView-LeftOf/RightOf (Linux) oder Horizontaler Bereich (Windows)
      //dazu ist das Window doppelt so breit wie die Aufloesung der Projection
      //3D TV:
      //eine Graka zeigt immer beiden Augen an, aber die Groesse fuer das
      //Window wird nicht verdoppelt wie bei einer GraKa und passiv stereo,
      //stattdessen gibt es 2 Channel die entsprechend konfiguriert werden
      if (stereoData == passive && bothEyesData)
      {
         width = 2 * width;
      }

      //active stereo
      bool stereo = false;
      if (stereoData == active)
      {
         stereo = true;
      }

      //Falls der render host gleich dem COVISE Gui host ist und einen
      //Control Monitor besitzt wird das projection window entsprechend
      //dem cMon3DOrientData oder der projection width mit left verschoben
      int left = 0;

      if (host == coviseGuiHostData && cMonData)
      {
         switch (cMon3DOrientData)
         {
            case leftOf:
            {
               left = cMonResData[0];
               break;
            }
            case rightOf:
            {
               left = width * -1;
               break;
            }
         }
      }

      configVrcw.localWindow(host, comment, width, height, stereo, left);
   }

   //COVISE GUI host == Master
   if (allHosts.size() != renderHosts.size())
   {
      //constants for the master
      const QString host = coviseGuiHostData;
      const QString comment = "MASTER";
      const bool stereo = false;
      const int left = 100;
      const int bottom = 100;
      const QString decoration = "true";

      //determine the fitting index in wallDim for Front wall
      int i = 0;
      while (wallDim[i]->wall != Front)
      {
         ++i;
      }

      //calculate width/height of window
      int projWidth = wallDim[i]->res[0];
      int projHeight = wallDim[i]->res[1];

      QVector<int> winRes = masterWinChanWH(projWidth, projHeight);

      int width = winRes[0];
      int height = winRes[1];

      //Front wall for Master
      configVrcw.localWindow(host, comment, width, height, stereo, left,
            bottom, decoration);
   }
}

//generate data for ChannelConfig and ViewportConfig entries
//for each host in configVrcw.xml
//
void VRCWFinal::genChannelViewportConfig()
{
   QList<wallVal*> wallDim;

   switch (kindData)
   {
      case Powerwall:
      case _3D_TV://_3D_TV is a untiled Powerwall
      {
         wallDim = pWallDimData;
         break;
      }
      case CAVE:
      {
         wallDim = caveWallDimData;
         break;
      }
   }

   //render hosts
   for (QList<QStringList>::size_type i=0; i < hostProjectionData.size(); ++i)
   {
      QString host = hostProjectionData[i][2];
      QString stereoMode = hostProjectionData[i][1];
      QString comment = hostProjectionData[i][0] % "_" % stereoMode % "-Eye";

      if (hostProjectionData[i][1] == "Both")
      {
         comment += "s";
      }

      //setzen des stereoMode falls abweichend von Left oder Right
      //gueltig fuer active, checkerboard, vInterlaced, cInterleave,
      //hInterlaced, rInterleave
      switch (stereoData)
      {
         case active:
         {
            stereoMode = "Quad_Buffer";
            break;
         }
         case checkerboard:
         {
            stereoMode = "Checkerboard";
            break;
         }
         case vInterlaced:
         case cInterleave:
         {
            stereoMode = "Vertical_Interlace";
            break;
         }
         case hInterlaced:
         case rInterleave:
         {
            stereoMode = "Horizontal_Interlace";
            break;
         }
         default:
         {
            //do nothing
            break;
         }
      }

      QStringList wallStrings = hostProjectionData[i][0].split(QRegExp("[-_]"),
            QString::SkipEmptyParts);
      cWall wall = strToCWall(wallStrings[0]);

      //determine the fitting index in caveWallDimData for wall
      int j = 0;
      while ( (wallDim.size() > 1) && (wall != wallDim[j]->wall) )
      {
         ++j;
      }

      //Resolution[Pixel]
      int width = wallDim[j]->res[0];
      int height = wallDim[j]->res[1];

      //falls das Seitenverhaeltnis der Wand nicht
      //dem der Aufloesung entspricht,
      //wird nicht die gesamte Aufloesung fuer die Darstellung benutzt
      //
      //Annahmen:
      //- die Wand ist breiter als hoch (als fuer Aufloesungs-Ratio notwendig)
      //- das Verhaeltnis Breite zu Hoehe ist bei der Aufloesung (Pixel)
      //  kleiner als bei der Abmessung (mm)
      //==> Breite der Aufloesung wird voll genutzt, Hoehe angepasst
      //
      //Annahmen:
      //- die Wand ist hoeher als breit (als fuer Aufloesungs-Ratio notwendig)
      //- das Verhaeltnis Breite zu Hoehe ist bei der Aufloesung (Pixel)
      //  groesser als bei der Abmessung (mm)
      //==> Hoehe der Aufloesung wird voll genutzt, Breite angepasst
      //

      //moegliche Anpassung der Breite an die voll genutzte Hoehe im CAVE
      //z.B fuer Boden/Decke, falls die Wand nicht dem Seitenverhaeltnis
      //der Aufloesung entspricht
      //bei Powerwall kann eine Anpassung ebenfalls noetig sein, falls eine
      //bestehende Wand mit Projektoren hoeherer Aufloesung als die alten
      //bestueckt wird, diese aber ein anderes Seitenverhaeltnis haben
      //als die Wand
      //bei _3D_TV eher unueblich
      int widthAdapted = 0;
      int heightAdapted = 0;

      //left:
      //Wert {Pixel], mit dem links und rechts die Darstellung verkleinert
      //wird
      //=> Haelfte der Differenz von voller Breite und angepasster Breite
      //
      //bottom:
      //Wert {Pixel], mit dem oben und unten die Darstellung verkleinert
      //wird
      //=> Haelfte der Differenz von voller Hoehe und angepasster Hoehe
      //
      //only necessary with CAVE and with Powerwall; with 3DTV normally not necessary
      //but possible because 3DTV is treated the same as Powerwall
      int left = 0;
      int bottom = 0;

      //Groesse der CAVE-Seiten [mm]
      //Groesse der Powerwall [mm]
      int screenSizeWidth = wallDim[j]->screenSize[0];
      int screenSizeHeight = wallDim[j]->screenSize[1];

      //Verhaeltnis Breite zu Hoehe der Abmessung der Screen [mm]
      double sizeRatio = screenSizeWidth * 1.0 / screenSizeHeight;

      //Verhaeltnis Breite zu Hoehe der Aufloesung [pixel]
      double resRatio = width * 1.0 / height;

      //Bestimmung der angepassten Breite (widthAdapted)
      //und Hoehe (heightAdapted) und left/bottom
      //nach obigen Annahmen
      //
      if (sizeRatio > resRatio)
      {
         //Verhaeltnis der Abmessung ist groesser als das der Aufloesung
         //=>Breite der Aufloesung wird voll genutzt, Hoehe wird angepasst
         widthAdapted = width;

         //max nutzbare Hoehe der Aufloesung ergibt sich aus der Breite der
         //Aufloesung und dem Seitenverhaeltnis der Abmessung der Wand
         //es wird immer der ganzzahlige kleinere Wert genommen
         int heightMaxUse = trunc(width / sizeRatio);

         //mit der Hoehe der Aufloesung und der max. nutzbaren Hoehe
         //(== heightMaxUse) ergibt sich der Offset bottom bei dem mit der
         //Darstellung begonnen wird
         //es wird immer der ganzzahlige kleinere Wert genommen, da nur um
         //vollstaendige Pixel verschoben werden kann
         bottom = trunc((height - heightMaxUse) / 2);

         //mit left kann die angepasste Breite berechnet werden
         heightAdapted = height - bottom * 2;
      }
      else if (sizeRatio < resRatio)
      {
         //Verhaeltnis der Abmessung ist kleiner als das der Aufloesung
         //=>Hoehe der Aufloesung wird voll genutzt, Breite wird angepasst
         heightAdapted = height;

         //max nutzbare Breite der Aufloesung ergibt sich aus der Hoehe der
         //Aufloesung und dem Seitenverhaeltnis der Abmessung der Wand
         //es wird immer der ganzzahlige kleinere Wert genommen
         int widthMaxUse = trunc(height * sizeRatio);

         //mit der Breite der Aufloesung und der max. nutzbaren Breite
         //(== widthMaxUse) ergibt sich der Offset left bei dem mit der
         //Darstellung begonnen wird
         //es wird immer der ganzzahlige kleinere Wert genommen, da nur um
         //vollstaendige Pixel verschoben werden kann
         left = trunc((width - widthMaxUse) / 2);

         //mit left kann die angepasste Breite berechnet werden
         widthAdapted = width - left * 2;
      }
      else
      {
         //Verhaeltnis der Aufloesung ist gleich der der Abmessung
         widthAdapted = width;
         heightAdapted = height;
         left = 0;
         bottom = 0;
      }

      //Bilder fuer rechtes und linkes Auge werden von einer Graka erzeugt:
      //bothEyeData == true
      //- passive: TwinView-LeftOf/RightOf (Linux)
      //           oder Horizontaler Bereich (Windows)
      //- bei active, sideBySide, topBottom, checkerboard, vInterlaced,
      //  cInterleave, hInterlaced, rInterleave ist immer bothEyeData == true
      if (bothEyesData)
      {
         for (int i = 0; i < (2 * grakaData); ++i)
         {
            int channelBE = i;
            int nameBE = i;
            int viewportIndexBE = i;
            int bottomBE = bottom;
            int leftBE = left;
            int windowIndexBE = trunc(i / 2);
            int widthAdaptedBE = 0;
            int heightAdaptedBE = 0;
            QString commentBE;
            QString stereoModeBE;

            //widthAdapted and heightAdapted are even values
            //because width and height of the resolution are even values
            //subtraction of an int * 2 (is even) from an even value is even

            //for topBottom and sideBySide we must have other left/bottom
            //and width/height
            int leftSbS = round(left / 2);
            int widthAdaptedSbS = width / 2 - leftSbS * 2;
            int bottomTb = round(bottom / 2);
            int heightAdaptedTb = height / 2 - bottomTb * 2;

            if (i % 2 == 0)
            {
               commentBE = comment % "_LEFT";
               stereoModeBE = "Left";

               switch (stereoData)
               {
                  case passive:
                  {
                     widthAdaptedBE = leftBE + widthAdapted;

                     heightAdaptedBE = bottomBE + heightAdapted;
                     break;
                  }
                  case active:
                  case checkerboard:
                  case vInterlaced:
                  case cInterleave:
                  case hInterlaced:
                  case rInterleave:
                  {
                     widthAdaptedBE = leftBE + widthAdapted;

                     heightAdaptedBE = bottomBE + heightAdapted;

                     stereoModeBE = stereoMode;
                     break;
                  }
                  case topBottom:
                  {
                     widthAdaptedBE = leftBE + widthAdapted;

                     bottomBE = bottomTb * 3 + heightAdaptedTb;
                     heightAdaptedBE = bottomBE + heightAdaptedTb;
                     break;
                  }
                  case sideBySide:
                  {
                     leftBE = leftSbS;
                     widthAdaptedBE = widthAdaptedSbS;

                     heightAdaptedBE = bottomBE + heightAdapted;
                     break;
                  }
               }
            }
            else
            {
               commentBE = comment % "_RIGHT";
               stereoModeBE = "Right";

               switch (stereoData)
               {
                  case passive:
                  {
                     leftBE = left + width;
                     widthAdaptedBE = leftBE + widthAdapted;

                     heightAdaptedBE = bottomBE + heightAdapted;
                     break;
                  }
                  case active:
                  case checkerboard:
                  case vInterlaced:
                  case cInterleave:
                  case hInterlaced:
                  case rInterleave:
                  {
                     widthAdaptedBE = leftBE + widthAdapted;

                     heightAdaptedBE = bottomBE + heightAdapted;

                     stereoModeBE = stereoMode;
                     break;
                  }
                  case topBottom:
                  {
                     widthAdaptedBE = leftBE + widthAdapted;

                     bottomBE = bottomTb;
                     heightAdaptedBE = heightAdaptedTb;
                     break;
                  }
                  case sideBySide:
                  {
                     leftBE = leftSbS * 3 + widthAdaptedSbS;
                     widthAdaptedBE = leftBE + widthAdaptedSbS;

                     heightAdaptedBE = bottomBE + heightAdapted;
                     break;
                  }
               }
            }

            configVrcw.localChannel(host, commentBE, stereoModeBE.toUpper(),
                  channelBE, windowIndexBE, viewportIndexBE, nameBE);
            configVrcw.localViewport(host, widthAdaptedBE, heightAdaptedBE, leftBE,
                  bottomBE, windowIndexBE, nameBE);
         }
      }
      else
      {
         configVrcw.localChannel(host, comment, stereoMode.toUpper());
         configVrcw.localViewport(host, widthAdapted, heightAdapted, left, bottom);
      }
   }

   //COVISE GUI host == Master
   if (allHosts.size() != renderHosts.size())
   {
      //constants for the master
      const QString host = coviseGuiHostData;
      const QString comment = "MASTER";
      const QString stereoMode = "MONO_MIDDLE";

      //determine the fitting index in wallDim for Front wall
      int i = 0;
      while (wallDim[i]->wall != Front)
      {
         ++i;
      }

      //calculate width/height of channel
      int projWidth = wallDim[i]->res[0];
      int projHeight = wallDim[i]->res[1];

      QVector<int> chanRes = masterWinChanWH(projWidth, projHeight);

      int width = chanRes[0];
      int height = chanRes[1];

      //Front wall for Master
      configVrcw.localChannel(host, comment, stereoMode);
      configVrcw.localViewport(host, width, height);
   }
}

//generate data for ScreenConfig entries for each host in configVrcw.xml
//
void VRCWFinal::genScreenConfig()
{
   QList<wallVal*> wallDim;

   switch (kindData)
   {
      case Powerwall:
      case _3D_TV://_3D_TV is a untiled Powerwall
      {
         wallDim = pWallDimData;
         break;
      }
      case CAVE:
      {
         wallDim = caveWallDimData;
         break;
      }
   }

   //render hosts
   for (QList<QStringList>::size_type i=0; i < hostProjectionData.size(); ++i)
   {
      QString host;
      QString comment;

      host = hostProjectionData[i][2];
      comment = hostProjectionData[i][0] % "_" %
            hostProjectionData[i][1] % "-Eye";

      if (hostProjectionData[i][1] == "Both")
      {
         comment += "s";
      }

      QStringList wallStrings = hostProjectionData[i][0].split(QRegExp("[-_]"),
            QString::SkipEmptyParts);
      cWall wall = strToCWall(wallStrings[0]);

      //determine the fitting index in caveWallDimData or pWallDimData for wall
      //wallDim == pWallDimData: wallDim.size == 1 and Index == 0
      int j = 0;
      while ( (wallDim.size() > 1) && (wall != wallDim[j]->wall) )
      {
         ++j;
      }

      //width and height of the Screen [mm]
      int width = wallDim[j]->screenSize[0];
      int height = wallDim[j]->screenSize[1];

      //Offset des Mittelpunkts der Screen
      //bezueglich der Mitte der CAVE-Seite
      //
      //wenn nur eine Wand betrachtet wird, hat jede Wand nur
      //eine x-Achse und eine y-Achse
      //offsetA in Richtung der Wand-x-Achse [mm]
      //offsetB in Richtung der Wand-y-Achse [mm]
      int offsetA = 0;
      int offsetB = 0;

      //dimension of whole wall if tiled
      int wsWidth = 0;
      int wsHeight = 0;

      if (tiledData)
      {
         //Abmessungen der ganzen Wand [mm]
         wsWidth = wallDim[j]->wallSize[0];
         wsHeight = wallDim[j]->wallSize[1];

         //Extract row and column number out of projection string
         //Row-1_Col-1 -> 1, 1
         //RegExp: \D+ : Matches one or more non-digit
         QStringList rcNum = hostProjectionData[i][0].split(QRegExp("\\D+"),
               QString::SkipEmptyParts);
         int rowNr = rcNum[0].toInt();
         int colNr = rcNum[1].toInt();

         //every wall has 2 dimensions:
         //width in x-direction (positive to the right),
         //height in y-direction (positive to the top)
         //
         //Row-1 and Col-1 are upper left side of the wall
         //Column count to the right (along positve x-axis)
         //Row count to the bottom (along negative y-axis)
         //Row-maxNo and Col-maxNo are lower right side of the wall
         switch (wallDim[j]->typeProj)
         {
            case Monitor:
            {
               int frameLR = wallDim[j]->frame[0];
               int frameBT = wallDim[j]->frame[1];

               offsetA = round((width - wsWidth) / 2) + frameLR
                     + (width + 2 * frameLR) * (colNr - 1);
               offsetB = round((wsHeight - height) / 2) - frameBT
                     - (height + 2 * frameBT) * (rowNr - 1);
               break;
            }
            case Projector:
            {
               int overlapHori = wallDim[j]->overlap[0];
               int overlapVert = wallDim[j]->overlap[1];

               offsetA = round((width - wsWidth) / 2)
                     + (width - overlapHori) * (colNr - 1);
               offsetB = round((wsHeight - height) / 2)
                     - (height - overlapVert) * (rowNr - 1);
               break;
            }
         }
      }

      //Koordinaten des Mittelpunkts der Screen [mm]
      int originX = 0;
      int originY = 0;
      int originZ = 0;

      //Rotationen um die Koordinatenachsen:
      //h: z-Achse;   p: x-Achse;   r: y-Achse
      double h = 0.0;
      double p = 0.0;
      double r = 0.0;

      switch (kindData)
      {
         case Powerwall://_3D_TV is a untiled Powerwall
         case _3D_TV://but a tiled _3D_TV isn't provided
         {
            originX = offsetA;
            originZ = offsetB;
            break;
         }
         case CAVE:
         {
            //Nullpunkt liegt in der Mitte des Quaders (Raumes)
            //von da aus werden die Waende entlang der Achsen verschoben
            //und um die Achsen gedreht
            //
            //Annahme:
            //-Left/Right und Bottom/Top sind mit Front verbunden
            //-Back hat die gleiche Groesse wie Front
            //-Left/Right haben die gleiche Hoehe wie Front
            //-Bottom/Top haben die gleiche Breite wie Front
            //
            //width of front wall is width of cave
            //depth of cave is largest value of left/right width
            //or bottom/top height
            //is to adjust with the specific dimension of left/right/bottom/top
            //height of front wall is height of cave
            int xVal = round(caveDimData[0] / 2);
            int yVal = round(caveDimData[1] / 2);
            int zVal = round(caveDimData[2] / 2);
            //Rotation in 90-Grad-Schritten im oder gegen Uhrzeigersinn
            double rot = 90.0;

            //originY is to adjust for left/right/bottom/top
            //with width/height for untiled
            //and wsWidth/wsHeight for tiled
            int adjWidth = 0;
            int adjHeight = 0;

            if (tiledData)
            {
               adjWidth = wsWidth;
               adjHeight = wsHeight;
            }
            else
            {
               adjWidth = width;
               adjHeight = height;
            }

            switch (wall)
            {
               case Front:
               {
                  originX = offsetA;
                  originY = yVal;
                  originZ = offsetB;
                  break;
               }
               case Left:
               {
                  originX = xVal * -1;
                  originY = offsetA  + (yVal - round(adjWidth / 2));
                  originZ = offsetB;
                  h = rot;
                  break;
               }
               case Right:
               {
                  originX = xVal;
                  originY = offsetA * -1 + (yVal - round(adjWidth / 2));
                  originZ = offsetB;
                  h = rot * -1;
                  break;
               }
               case Bottom:
               {
                  originX = offsetA;
                  originY = offsetB + (yVal - round(adjHeight / 2));
                  originZ = zVal * -1;
                  p = rot * -1;
                  break;
               }
               case Top:
               {
                  originX = offsetA;
                  originY = offsetB * -1 + (yVal - round(adjHeight / 2));
                  originZ = zVal;
                  p = rot;
                  break;
               }
               case Back:
               {
                  originX = offsetA * -1;
                  originY = yVal * -1;
                  originZ = offsetB;
                  h = rot * 2;
                  break;
               }
            }
            break;
         }
      }

      //Korrektur fuer Nullpunkt
      //Korrektur von originX, originY, originZ
      //falls Nullpunkt nicht in der Mitte der Cave dimension
      //sondern Mitte Front oder Mitte Bottom ist
      if (kindData == CAVE)
      {
         //if zero point in middle of Cave dimension this function do nothing
         QVector<int> origin = zeroPointCorrection(originX, originY, originZ);

         originX = origin[0];
         originY = origin[1];
         originZ = origin[2];
      }

      //Bilder fuer rechtes und linkes Auge werden von einer Graka erzeugt:
      //bothEyeData == true
      //- passive: TwinView-LeftOf/RightOf (Linux)
      //           oder Horizontaler Bereich (Windows)
      //-  bei active, sideBySide, topBottom, checkerboard, vInterlaced,
      //  cInterleave, hInterlaced, rInterleave ist immer bothEyeData == true
      if (bothEyesData)
      {
         for (int i = 0; i < (2 * grakaData); ++i)
         {
            int nameBE = i;
            int screenBE = i;
            QString commentBE;

            //Ergaenzen des Kommentars mit 2 Moeglichkeiten
            if (i % 2 == 0)
            {
               commentBE = comment % "_LEFT";
            }
            else
            {
               commentBE = comment % "_RIGHT";
            }

            configVrcw.localScreen(host, commentBE, width, height,
               originX, originY, originZ, h, p, r, nameBE, screenBE);
         }
      }
      else
      {
         configVrcw.localScreen(host, comment, width, height,
            originX, originY, originZ, h, p, r);
      }
   }

   //COVISE GUI host == Master
   if (allHosts.size() != renderHosts.size())
   {
      //constants for the master
      const QString host = coviseGuiHostData;
      const QString comment = "MASTER";

      int width, height;

      //determine the fitting index in wallDim for Front wall
      int i = 0;
      while (wallDim[i]->wall != Front)
      {
         ++i;
      }

      if (tiledData)
      {
         width = wallDim[i]->wallSize[0];
         height = wallDim[i]->wallSize[1];
      }
      else
      {
         width = wallDim[i]->screenSize[0];
         height = wallDim[i]->screenSize[1];
      }

      //calculated variable
      int originX = 0;
      int originY = 0;
      int originZ = 0;

      switch (kindData)
      {
         case CAVE:
         {
            //zero point is in the middle of Cave dimension
            //front wall is connected to this depth value
            //zero point is in the middle of this cave volume
            originY = round(caveDimData[1] / 2);

            //correction for zero point if in Front wall or Bottom wall
            //if zero point in middle of Cave dimension this function do nothing
            QVector<int> origin = zeroPointCorrection(originX, originY,
                  originZ);

            originX = origin[0];
            originY = origin[1];
            originZ = origin[2];

            break;
         }
         case Powerwall://_3D_TV is a untiled Powerwall
         case _3D_TV://but a tiled _3D_TV isn't provided
         {
            //do nothing
            break;
         }
      }

      //Front Config for Master
      configVrcw.localScreen(host, comment, width, height, originX, originY,
            originZ);
   }
}

//generate data for Input entry in configVrcw.xml
//
void VRCWFinal::genInput()
{
   //used in replacement in hand/head sensor, buttons, buttonDevice labels
   const QString SENSOR = " sensor ";
   const QString REPLACE = "";
   const QString HAND = "Hand";
   const QString BUTTON = "Button";
   const QString BTNDEV = "ButtonDev";
   //Label for Head Sensor 0 == ConstHead
   const QString cHEAD = "ConstHead";
   //Label for Device TrackingSystem
   const QString tSYSTEM = "TrackingSystem";

   int handSensNum = thwvdData->numHandSens;
   int headSensNum = thwvdData->numHeadSens;
   int personsNum = thwvdData->numPersons;


   //basic input
   //
   configVrcw.inputBasic();


   //Tracking System
   //
   //tSysName   -->> "TrackingSystem"
   //tSysDriver   -->> thwvdData->tSys umwandeln
   // opt. tSysHost and/or tSysPort   -->> thwvdData->...
   //tSysOffset   -->> thwvdData->tSysDim
   //tSysOrient   -->> thwvdData->tSysDim

   //Driver for TrackingSystem
   QString trackSysDrv;
   switch (thwvdData->tHandle)
   {
      case cov:
      {
         switch (thwvdData->tSys)
         {
            case ART:
            {
               trackSysDrv = "dtrack";
               break;
            }
            case Vicon:
            {
               trackSysDrv = "tarsus";
               break;
            }
            case InterSense://shouldn't be available here but it's a Polhemus
            case Polhemus:
            {
               trackSysDrv = "polhemus";
               break;
            }
            case FOB:
            {
               trackSysDrv = "fob";
               break;
            }
            case Motionstar:
            {
               trackSysDrv = "motionstar";
               break;
            }
            case Wii:
            {
               trackSysDrv = "wii";
               break;
            }
         }
         break;
      }
      case vrc:
      {
         trackSysDrv = "vrc";
         break;
      }
   }

   configVrcw.inputTrackSysDev(tSYSTEM, trackSysDrv, thwvdData->tSysDim);

   //additional entries for specific trackingSystems
   switch (thwvdData->tHandle)
   {
      case cov:
      {
         switch (thwvdData->tSys)
         {
            case ART:
            {
               configVrcw.addAttrToElemInDev(tSYSTEM, "port",
                     QString::number(thwvdData->artRPort));

               if (thwvdData->checkArtHost)
               {
                  configVrcw.addAttrToElemInDev(tSYSTEM, "host", thwvdData->hostIPAddr);
                  configVrcw.addAttrToElemInDev(tSYSTEM, "startup",
                     QString::number(thwvdData->artHostSPort));
               }
               break;
            }
            case Vicon:
            {
               configVrcw.addAttrToElemInDev(tSYSTEM, "host", thwvdData->hostIPAddr);
               break;
            }
            case Motionstar:
            {
               configVrcw.addAttrToElemInDev(tSYSTEM, "ipaddress", thwvdData->hostIPAddr);
               break;
            }
            case InterSense://shouldn't be available here but it's a Polhemus
            case Polhemus:
            case FOB: //FOB and Polhemus are here the same
            {
               configVrcw.addAttrToElemInDev(tSYSTEM, "serialport",
                  thwvdData->polFobSPort);
               break;
            }
            case Wii:
            {
               //do nothing
               break;
            }
         }
         break;
      }
      case vrc:
      {
         trackSysDrv = "vrc";
         configVrcw.addAttrToElemInDev(tSYSTEM, "port",
            QString::number(thwvdData->vrcPort));
         break;
      }
   }



   //Button Device
   //
   //jeder HandSensor hat sein eigenes ButtonDevice, somit
   //HandSensor_1 -> ButtonDev1
   //HandSensor_2 -> ButtonDev2
   for (int i = 0; i < handSensNum; ++i)
   {
      QString btnDevName =
            thwvdData->handSVal[i]->sensLabel.replace(SENSOR, REPLACE);
      btnDevName = btnDevName.replace(HAND, BTNDEV);
      QString dev = thwvdData->handSVal[i]->bDev;

      //driver for ButtonSystem specified in handSVal[i]->bDrv
      QString btnDevDrv;
      switch (thwvdData->tHandle)
      {
         case cov:
         {
            switch (thwvdData->handSVal[i]->bDrv)
            {
               case DTrack:
               {
                  btnDevDrv = "dtrack";
                  break;
               }
               case Mouse_Buttons:
               {
                  btnDevDrv = "mousebuttons";
                  break;
               }
               case PolhemusDrv:
               {
                  btnDevDrv = "polhemus";
                  configVrcw.addAttrToElemInDev(tSYSTEM, "inputdevice",
                        btnSysToStr(thwvdData->handSVal[i]->bSys));
                  break;
               }
               case FOB_Drv:
               {
                  btnDevDrv = "fob";
                  break;
               }
               case MotionstarDrv:
               {
                  btnDevDrv = "motionstar";
                  break;
               }
               case HornetDrv:
               {
                  btnDevDrv = "hornet";
                  break;
               }
               case MikeDrv:
               {
                  btnDevDrv = "mike";
                  break;
               }
               case WiimoteDrv:
               {
                  btnDevDrv = "wiimote";
                  break;
               }
            }

            configVrcw.inputButtonDev(btnDevName, btnDevDrv, dev);
            break;
         }
         case vrc:
         {
            btnDevDrv = "vrc";

            configVrcw.inputButtonDev(btnDevName, btnDevDrv);

            //add optional button address if vrc
            switch (thwvdData->handSVal[i]->bSys)
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
                  configVrcw.addAttrToElemInDev(btnDevName, "buttonAddress",
                        QString::number(thwvdData->handSVal[i]->bAddr));
                  break;
               }
            }
            break;
         }
      }
   }



   //Body
   //
   //Hand
   for (int i = 0; i < handSensNum; ++i)
   {
      QString bName = thwvdData->handSVal[i]->sensLabel.replace(SENSOR, REPLACE);
      sensTrackSysDim* hhSensDim = thwvdData->handSDim[i];
      QString dev = tSYSTEM;
      int bIndex = thwvdData->handSVal[i]->bIndex;

      configVrcw.inputBody(bName, hhSensDim, dev, bIndex);
   }

   //ConstHead
   QString bName = cHEAD;
   sensTrackSysDim* constHeadDim = new sensTrackSysDim();

   if (thwvdData->headSDim[0]->x == 0 && thwvdData->headSDim[0]->y == 0
         && thwvdData->headSDim[0]->z == 0 && thwvdData->headSDim[0]->h == 0
         && thwvdData->headSDim[0]->p == 0 && thwvdData->headSDim[0]->r == 0)
   {
      //default values
      constHeadDim->x = 0;
      constHeadDim->y = -2000;
      constHeadDim->z = 0;
   }
   else
   {
      constHeadDim = thwvdData->headSDim[0];
   }

   configVrcw.inputBody(bName, constHeadDim);

   //Head
   //Head Sensor 0 is ConstHead
   for (int i = 1; i <= headSensNum; ++i)
   {
      QString bName = thwvdData->headSVal[i]->sensLabel.replace(SENSOR, REPLACE);
      sensTrackSysDim* hhSensDim = thwvdData->headSDim[i];
      QString dev = tSYSTEM;
      int bIndex = thwvdData->headSVal[i]->bIndex;

      configVrcw.inputBody(bName, hhSensDim, dev, bIndex);
   }



   //Buttons
   //
   //fuer jeden HandSensor gibt es ein ButtonDevice und somit auch
   //je einen Eintrag fuer die Buttons
   for (int i = 0; i < handSensNum; ++i)
   {
      QString hName = thwvdData->handSVal[i]->sensLabel.replace(SENSOR, REPLACE);
      QString btnName = hName;
      btnName = btnName.replace(HAND, BUTTON);
      QString btnDevName = hName;
      btnDevName = btnDevName.replace(HAND, BTNDEV);

      configVrcw.inputButtons(btnName, btnDevName);
   }



   //Persons
   //
   for (int i = 0 ; i < personsNum; ++i)
   {
      QString pName = "Tracked" + REPLACE + QString::number(i + 1);
      QString hand = thwvdData->persVal[i]->handSens.replace(SENSOR, REPLACE);
      QString head = thwvdData->persVal[i]->headSens.replace(SENSOR, REPLACE);

      if (head == "Head" + REPLACE + "0")
      {
         head = cHEAD;
      }

      QString button = hand;
      button = button.replace("Hand","Button");

      configVrcw.inputPersons(pName, hand, head, button);
   }
}

//generate list of all configured hostnames
//(from hostProjectionData and coviseGuiHost)
//
void VRCWFinal::allHostnames()
{
   allHosts.clear();

   //Liste der render hosts generieren
   renderHostnames();

   allHosts = renderHosts;

   //coviseGuiHost zu allHosts am Anfang hinzufuegen
   //zuerst loeschen, falls enthalten
   if (allHosts.contains(coviseGuiHostData))
   {
      allHosts.removeAll(coviseGuiHostData);
   }
   allHosts.prepend(coviseGuiHostData);
}

//generate list of render hosts (from hostProjectionData)
//
void VRCWFinal::renderHostnames()
{
   renderHosts.clear();

   //RenderHosts aus der Liste extrahieren
   for (QList<QStringList>::size_type i = 0;
         i < hostProjectionData.size(); ++i)
   {
      renderHosts.append(hostProjectionData[i][2]);
   }
}

//add newlines into the output of the XMLConfigWriter class
//
QString VRCWFinal::addNlToStrOut(const QString& strToEdit) const
{
   QStringList strList = strToEdit.split('\n');

   QString result;
   result.append(strList[0]);
   result.append("\n\n");

   for (QStringList::size_type i = 1; i < strList.size(); ++i)
   {
      if (strList[i].contains(QRegExp("<[a-zA-Z]+>"))
            && !strList[i - 1].contains(QRegExp("</[a-zA-Z]+>")))
      {
         result.append('\n');
         result.append(strList[i]);
      }
      else if (strList[i].contains(QRegExp("</[a-zA-Z]+>")))
      {
         result.append(strList[i]);
         result.append('\n');
      }
      else
      {
         result.append(strList[i]);
      }

      if (strList[i].contains(QRegExp("(<Input>|<COVER>)"))
            && !strList[i + 1].contains(QRegExp("<[a-zA-Z]+>")))
      {
         result.append('\n');
      }

      if (strList[i].contains(QRegExp("<!--"))
            && !strList[i + 1].contains(QRegExp("<!--")))
      {
         result.append('\n');
      }

      result.append('\n');
   }

   return result;
}

//calculate window width/height on master (coviseGuiHost)
//
QVector<int> VRCWFinal::masterWinChanWH(const int& projWidth, const int& projHeight) const
{
   QVector<int> masterRes(2,0);

   //calculated variables
   int width, height;

   //minimal width for Window/Channel is half width of control monitor
   //or at least 800 Pixel
   const int cMonWidth = cMonResData[0];
   const int hCMonWidth = round(cMonWidth / 2);
   const int minWidth = 800;

   //calc the width of the window/channel
   if (hCMonWidth < minWidth)
   {
      width = minWidth;
   }
   else
   {
      width = hCMonWidth;
   }

   //calc the height of the window/channel with the aspect ratio
   //of the 3D Projection
   height = round(width * projHeight / projWidth);

   masterRes[0] = width;
   masterRes[1] = height;

   return masterRes;
}

//correction for zeroPoint if in Front or Bottom
//
QVector<int> VRCWFinal::zeroPointCorrection(const int& originX,
      const int& originY, const int& originZ) const
{
   int xOrigin = originX;
   int yOrigin = originY;
   int zOrigin = originZ;

   switch (zeroPData)
   {
      case fWall:
      {
         //height of Front wall is height in Cave dimension
         //width of Front wall is width of Cave dimension
         //so only originY must be corrected
         //yVallCorrFWall is connected to caveDimData[1]
         //(the largest dimension of left/right width
         //or bottom/top height)

         //originY
         int yValCorrFWall = round(caveDimData[1] / 2);
         yOrigin = originY - yValCorrFWall;

         break;
      }
      case boWall:
      {
         //width of bottom wall is width of Cave dimension
         //originY and originZ must be corrected

         //originY
         //determine the fitting index in caveWallDimData for Bottom wall
         int i = 0;
         while (caveWallDimData[i]->wall != Bottom)
         {
            ++i;
         }
         QVector<int> bottomWallSize;

         if (tiledData)
         {
            bottomWallSize = caveWallDimData[i]->wallSize;
         }
         else
         {
            bottomWallSize = caveWallDimData[i]->screenSize;
         }

         int caveDepth = caveDimData[1];
         int yValCorrBoWall = round((caveDepth - bottomWallSize[1]) / 2);
         yOrigin = originY - yValCorrBoWall;

         //originZ
         int zValCorrBoWall = round(caveDimData[2] / 2);
         zOrigin = originZ + zValCorrBoWall;

         break;
      }
      case cDim:
      {
         //do nothing
         break;
      }
   }

   QVector<int> origin(3);
   origin[0] = xOrigin;
   origin[1] = yOrigin;
   origin[2] = zOrigin;

   return origin;
}

bool VRCWFinal::save()
{
   //configUser
   //
   QFile fileUser(confStorPath + confNameUser);

   //try to open file for writing
   if (!fileUser.open(QIODevice::WriteOnly | QIODevice::Text))
   {
      QString messageUser = tr("Unable to open ") + confStorPath
            + confNameUser + tr(" for writing.");
      QMessageBox::warning(this, tr("Error"), messageUser + "\n\n"
            + fileUser.errorString());

      return false;
   }

   //write content of configUserTextEdit as plainText into fileUser
   QTextStream outUser(&fileUser);
   outUser << ui.configUserTextEdit->toPlainText();

   //set modified of configVrcwTextEdit and call configTextChanged()
   ui.configUserTextEdit->document()->setModified(false);
   configTextChanged();

   //configVrcw
   //
   QFile fileVrcw(confStorPath + confNameVrcw);

   //try to open file for writing
   if (!fileVrcw.open(QIODevice::WriteOnly | QIODevice::Text))
   {
      QString messageVrcw = tr("Unable to open ") + confStorPath
            + confNameVrcw + tr(" for writing.");
      QMessageBox::warning(this, tr("Error"), messageVrcw + "\n\n"
            + fileVrcw.errorString());

      return false;
   }

   //write content of configUserTextEdit as plainText into fileVrcw
   QTextStream outVrcw(&fileVrcw);
   outVrcw << ui.configVrcwTextEdit->toPlainText();

   return true;
}



/*****
 * private slots
 *****/

//disable Edit button for configVrcw.xml
//
void VRCWFinal::roConfigVrcwTextEdit(const int& index) const
{
   //index = 0: configUser
   //index = 1: configVrcw: should be read only

   switch (index)
   {
      case 0:
      {
         ui.EditPushButton->setEnabled(true);

         if (ui.configUserTextEdit->isReadOnly())
         {
            ui.EditPushButton->setText("Edit");
         }
         else
         {
            ui.EditPushButton->setText("Read Only");
         }

         break;
      }
      case 1:
      {
         ui.configVrcwTextEdit->setReadOnly(true);
         ui.EditPushButton->setText("Edit");
         ui.EditPushButton->setDisabled(true);
         break;
      }
   }
}

//Edit button
//
void VRCWFinal::editReadOnly() const
{
   //index = 0: configUser: editable
   //index = 1: configVrcw: should be read only

   if (ui.configTabWidget->currentIndex() == 0)
   {
      if (ui.configUserTextEdit->isReadOnly())
      {
         ui.configUserTextEdit->setReadOnly(false);
         ui.EditPushButton->setText("Read Only");
      }
      else
      {
         ui.configUserTextEdit->setReadOnly(true);
         ui.EditPushButton->setText("Edit");
      }
   }
}

//Save button
//
bool VRCWFinal::save_exec()
{
   if (save())
   {
      //open messageBox if files saved successful
      QString messageSuccess = tr("Config files ") + confNameUser + tr(" and ")
               + confNameVrcw + tr(" saved successfully in\n") + confStorPath;
      QMessageBox::information(this, tr("Success"), messageSuccess);

      return true;
   }
   else
   {
      return false;
   }
}

//check if the textEditWidget is modified
//
void VRCWFinal::configTextChanged()
{
   //modified state of configUsetextEdit
   //configVrcwTextEdit is always read only
   bool confModifiedNew = ui.configUserTextEdit->document()->isModified();

   if (confModified != confModifiedNew)
   {
      confModified = confModifiedNew;

      emit configIsModified(confModified);
   }
}
