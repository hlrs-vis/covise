#ifndef VRCWFINAL_H
#define VRCWFINAL_H

#include <QWidget>
#include "ui_vrcwfinal.h"

#include "vrcwbase.h"
#include "xmlconfigwriter.h"
#include "vrcwutils.h"

class configVal;
class wallVal;
class trackHwValDim;


class VRCWFinal : public QWidget, public VRCWBase
{
    Q_OBJECT

public:
   /*****
    * constructor - destructor
    *****/
   VRCWFinal(QWidget* parent = 0);
   ~VRCWFinal();


   /*****
    * functions
    *****/
   //Entgegennehmen des Speicherortes und des Namens fuer
   //die zu erzeugende configUser.xml und configVrcw.xml
   void setstart(configVal*& coVal);

   //Entgegennehmen von ProjectionKind, Tiled, Stereo, bothEyes, No of Graka
   //Number of Rows/Columns, List of CaveWalls von ProjectionHW
   void setProjectionHw(const proKind& kind, const bool& tiled,
         const stType& stereo, const bool& bothEyes, const int& graka,
         const bool& cMon, const QVector<int>& cMonRes,
         const loRoOrient& cMon3DOrient);

   //Entgegennehmen von viewerPos, floorHeight und Bezeichnung der wall,
   //resolution, size, rowCol, typeProj, wallSize, overlap und frame
   //von VRCWProjectionDimPowerwall
   void setProjectDimPowerwall(const QVector<int>& vPosPw, const int& floorPw,
         const QList<wallVal*>& pWallDim);

   //Entgegennehmen von zeroPoint viewerPos, floorHeight, Cave-Abmessungen
   //und Bezeichnung der wall, resolution, size, rowCol, typeProj, wallSize,
   //overlap und frame fuer die einzelnen Waende von VRCWProjectionDimCave
   void setProjectDimCave(const zPoint& zeroP, const QVector<int>& vPosC,
         const int& floorC,
         const QVector<int>& caveDim, const QList<wallVal*>& caveWallDim);

   //Entgegennehmen von OS und executionMode von VRCWHost
   void setHost(const opSys& os, const execMode& exec);

   //Entgegennehmen von coviseGui und hostProjection von VRCWHostProjection
   void setHostProjection(const QString& coviseGuiHost,
         const QList<QStringList>& hostProjection);

   //Entgegennehmen von tracking system, button system, button device,
   //button driver, num of hand/head sensors, body index, num persons,
   //tracked persons
   //Entgegennehmen von Tracking, Hand und Head Offset und Orientation
   void setTrackingValDim(trackHwValDim*& thwvd);

   //Generierung der configs
   void createXmlConfigs();

   //Generierung der configUser.xml
   void createXmlConfigUser();

   //Generierung der configVrcw.xml
   void createXmlConfigVrcw();

   //Save Config
   bool saveConfig();


private:
   /*****
    * GUI Elements
    *****/
   Ui::VRCWFinalClass ui;


   /*****
    * functions
    *****/
   //generate data for SceneSize, ViewerPos and floorHeight in configVrcw.xml
   void genSceneViewerFloor();

   //generate data for Num{Pipes,Windows,Screens} and Stereo in configVrcw.xml
   void genNumPipesWindowsScreensStereo();

   //generate data for MultiPC in configVrcw.xml
   void genMultiPC();

   //generate data for PipeConfig entries for each host in configVrcw.xml
   void genPipeConfig();

   //generate data for WindowConfig entries for each host in configVrcw.xml
   void genWindowConfig();

   //generate data for ChannelConfig and ViewportConfig entries
   //for each host in configVrcw.xml
   void genChannelViewportConfig();

   //generate data for ScreenConfig entries for each host in configVrcw.xml
   void genScreenConfig();

   //generate data for Input entry in configVrcw.xml
   void genInput();

   //generate list of all configured hostnames
   void allHostnames();

   //generate list of render hosts
   void renderHostnames();

   //add newlines into the output of the XMLConfigWriter class
   QString addNlToStrOut(const QString& strToEdit) const;

   //calculate window width/height on master (coviseGuiHost)
   QVector<int> masterWinChanWH(const int& projWidth,
         const int& projHeight) const;

   //correction for zeroPoint if in Front or Bottom
   QVector<int> zeroPointCorrection(const int& originX, const int& originY,
         const int& originZ) const;

   //save config files
   bool save();



   /*****
    * variables
    *****/
   XMLConfigWriter configUser;
   XMLConfigWriter configVrcw;
   QString confNameUser;
   QString confNameVrcw;
   QString confStorPath;
   QStringList allHosts;
   QStringList renderHosts;
   bool confModified;

   // Daten aus vorherigen ConfigTabs
   //start
   configVal* coValData;
   //ProjectionHw
   proKind kindData;
   bool tiledData;
   stType stereoData;
   bool bothEyesData;
   int grakaData;
   bool cMonData;
   QVector<int> cMonResData;
   loRoOrient cMon3DOrientData;
   //ProjectionDimPowerwall
   QVector<int> vPosPwData;
   int floorPwData;
   QList<wallVal*> pWallDimData;
   //ProjectionDimCave
   zPoint zeroPData;
   QVector<int> vPosCData;
   int floorCData;
   QVector<int> caveDimData;
   QList<wallVal*> caveWallDimData;
   //host
   opSys osData;
   execMode execData;
   //HostProjection
   QString coviseGuiHostData;
   QList<QStringList> hostProjectionData;
   //TrackingHw und TrackingDim
   trackHwValDim* thwvdData;


signals:
   void configIsModified(const bool& modified);


private slots:
   void roConfigVrcwTextEdit(const int& index) const;
   //Buttons
   void editReadOnly() const;
   bool save_exec();

   //textEdit is modified
   void configTextChanged();

};

#endif // VRCWFINAL_H
