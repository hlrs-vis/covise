#ifndef XMLCONFIGWRITER_H_
#define XMLCONFIGWRITER_H_

#include <QDomDocument>
#include <QTextCodec>
#include <QStringList>
#include <QVector>

#include "vrcwutils.h"

class sensTrackSysDim;


class XMLConfigWriter
{
public:
   /*****
    * constructor - destructor
    *****/
   XMLConfigWriter();
   ~XMLConfigWriter();


   /*****
    * functions
    *****/
   //Initialize the domDocument with some Elements
   void initializeDomDocVrcw();

   //generate SceneSize, ViewerPos and FloorHeight entries in config.xml
   void sceneViewerFloor(const int& scene, const QVector<int>& vPos,
         const int& floor);

   //generate Num{Pipe,Windows,Screens,Stereo} entries in config.xml
   void numPipesWindowsScreensStereo(const int& pipes, const int& windows,
         const int& screens, const QString& stereoMode = "on");

   //generate MultiPC entry in config.xml
   void multiPC(const opSys& os, const execMode& exec,
         const QString& covGuiHost, const QStringList& hosts);

   //generate Local PipeConfig entry in config.xml
   void localPipe(const QString& host);

   //generate Local WindowConfig entry in config.xml
   void localWindow(const QString& host, const QString& comment,
         const int& width, const int& height, const bool& stereo,
         const int& left = 0, const int& bottom = 0,
         const QString& decoration = "false", const int& window = 0,
         const int& pipeIndex = 0, const int& name = 0);

   //generate Local ChannelConfig entry in config.xml
   void localChannel(const QString& host, const QString& comment,
         const QString& stereoMode, const int& channel = 0,
         const int& windowIndex = 0, const int& viewportIndex = 0,
         const int& name = 0);

   //generate Local ViewportConfig entry in config.xml
   void localViewport(const QString& host, const int& width,
         const int& height, const int& left = 0, const int& bottom = 0,
         const int& windowIndex = 0, const int& name = 0);

   //generate Local Screen entry in config.xml
   void localScreen(const QString& host, const QString& comment,
         const int& width, const int& height,
         const int& originX = 0, const int& originY = 0,
         const int& originZ = 0,
         const double& h = 0.0, const double& p = 0.0, const double& r = 0.0,
         const int& name = 0, const int& screen = 0);

   //generate Local Input basic entry in config.xml
   void inputBasic();

   //generate Local Input Device TrackingSys entry with name tSysName
   //in config.xml
   void inputTrackSysDev(const QString& tSysName, const QString& drv,
         sensTrackSysDim*& trackSysDim);

   //generate Local Input Device ButtonDev with name bDevName entry
   //in config.xml
   void inputButtonDev(const QString& bDevName, const QString& drv,
         const QString& dev = "");

   //adds attribute to Element with name elem in Local Input Device section
   void addAttrToElemInDev(const QString& elem, const QString& attrName,
         const QString& attrVal);

   //generate Local Input Body entry with name bName in config.xml
   void inputBody(const QString& bName, sensTrackSysDim*& hhSensDim,
         const QString& dev = "", const int& bIndex = -1);

   //generate Local Input Buttons entry with name btnsName in config.xml
   void inputButtons(const QString& btnsName, const QString& btnDev);

   //generate Local Input Persons entry in config.xml
   void inputPersons(const QString& pName, const QString& hand,
         const QString& head, const QString& button);

   //Output of domDocVrcw
   QString strOutVrcw(const int& indent = 3,
         QTextCodec* codec = QTextCodec::codecForName("ISO-8859-1")) const;

   //generate domDocUser for configUser.xml
   void generateConfigUser(const QString& coConfNameUser);

   //Output of domDocUser
   QString strOutUser(const int& indent = 3,
         QTextCodec* codec = QTextCodec::codecForName("ISO-8859-1")) const;


private:
   /*****
    * functions
    *****/
   //find or create QDomElement LOCAL with attribute host
   QDomElement findCreateLocalHostElem(const QString& host);

   //find or create special QDomElement specElemStr as child of a
   //specified QDomElement element
   QDomElement findCreateSpecElem(QDomElement& element,
         const QString& specElemStr, const char& insParam = 'a');

   //find or create special QDomElement specElemStr as child of
   //a COVER QDomElement of a Local QDomElement for specified host
   QDomElement findCreateLocalHostCoverSpecElem(const QString& host,
         const QString& specElemStr, const char& insParam = 'a');


   /*****
    * variables
    *****/
   //xml Document and Elements
   QDomDocument domDocUser;
   QDomDocument domDocVrcw;
   QDomElement coConf;
   QDomElement glob;
   QDomElement gCover;
   QDomNode xmlNode;

};

#endif /* XMLCONFIGWRITER_H_ */
