#include "xmlconfigwriter.h"

#include <QTextStream>
#include <QStringBuilder>

#include "datatypes.h"


/*****
 * constructor - destructor
 *****/

XMLConfigWriter::XMLConfigWriter()
{
   domDocUser = QDomDocument();
   domDocVrcw = QDomDocument();
}

XMLConfigWriter::~XMLConfigWriter()
{

}


/*****
 * public functions
 *****/

//Initialize the domDocument with some Elements
//
void XMLConfigWriter::initializeDomDocVrcw()
{
   domDocVrcw.clear();

   coConf = domDocVrcw.createElement("COCONFIG");
   coConf.setAttribute("version", "1");
   domDocVrcw.appendChild(coConf);

   glob = findCreateSpecElem(coConf, "GLOBAL");

   gCover = findCreateSpecElem(glob, "COVER");

   xmlNode = domDocVrcw.createProcessingInstruction("xml",
         "version=\"1.0\"");
   domDocVrcw.insertBefore(xmlNode, domDocVrcw.firstChild());
}

//generate SceneSize, ViewerPos and FloorHeight entries in config.xml
//
void XMLConfigWriter::sceneViewerFloor(const int& scene,
      const QVector<int>& vPos, const int& floor)
{
   QDomElement sceneSize = findCreateSpecElem(gCover, "SceneSize");
   sceneSize.setAttribute("value", scene);

   QDomElement viewerPos = findCreateSpecElem(gCover, "ViewerPosition");
   viewerPos.setAttribute("x", vPos[0]);
   viewerPos.setAttribute("y", vPos[1]);
   viewerPos.setAttribute("z", vPos[2]);

   QDomElement floorHeight = findCreateSpecElem(gCover, "FloorHeight");
   floorHeight.setAttribute("value", floor);
}

//generate Num{Pipe,Windows,Screens} entries in der config.xml
//
void XMLConfigWriter::numPipesWindowsScreensStereo(const int& pipes,
      const int& windows, const int& screens, const QString& stereoMode)
{
   QDomElement numPipes = findCreateSpecElem(gCover, "NumPipes");
   numPipes.setAttribute("value", pipes);

   QDomElement numWindows = findCreateSpecElem(gCover, "NumWindows");
   numWindows.setAttribute("value", windows);

   QDomElement numScreens = findCreateSpecElem(gCover, "NumScreens");
   numScreens.setAttribute("value", screens);

   QDomElement stereo = findCreateSpecElem(gCover, "Stereo");
   stereo.setAttribute("enable", stereoMode);
   stereo.setAttribute("value", stereoMode);
}

//generate MultiPC entry in config.xml
//
void XMLConfigWriter::multiPC(const opSys& os, const execMode& exec,
      const QString& covGuiHost, const QStringList& hosts)
{
   QDomElement multiPC = findCreateSpecElem(gCover, "MultiPC");

   QDomElement sync = findCreateSpecElem(multiPC, "SyncMode");
   sync.setAttribute("value", "TCP");

   int slaves = hosts.size();
   QDomElement numSlaves = findCreateSpecElem(multiPC, "NumSlaves");
   numSlaves.setAttribute("value", slaves);

   QDomElement masterIFace = findCreateSpecElem(multiPC, "MasterInterface");
   masterIFace.setAttribute("value", covGuiHost);

   for (QStringList::size_type i = 0; i < slaves; ++i)
   {
      QString valueString;

      switch (exec)
      {
         case ssh:
         {
            valueString = "ssh ${USER}@" % hosts[i] % " " %
                  "startOpenCOVER `pwd` $ARCHSUFFIX $COVISEDIR $COCONFIG";
            break;
         }
         case covremote:
         {
            switch (os)
            {
               case Linux:
               {
                  valueString = "covremote ${USERNAME}@" % hosts[i] %
                        " setenv DISPLAY :0.0\\;" %
                        " if \\( \\{ test -d `pwd` \\} \\)" %
                        " cd `pwd` \\; opencover \\>\\& /tmp/errors.log";
                  break;
               }
               case Windows:
               {
                  valueString = "covremote OpenCOVER " %
                        QString::number(i + 1)  % ";";
                  break;
               }
            }
            break;
         }
         case CovDaemon:
         {
            valueString = "CovDaemon OpenCOVER " %
                  QString::number(i + 1) % ";";
            break;
         }
         case rsh:
         {
            valueString = "rsh -l ${USER} " % hosts[i] % " " %
                  "startOpenCOVER `pwd` $ARCHSUFFIX $COVISEDIR $COCONFIG";
            break;
         }
      }

      QDomElement startup = domDocVrcw.createElement("Startup");
      startup.setAttribute("value", valueString);
      startup.setAttribute("name", i);
      multiPC.appendChild(startup);
   }
}

//generate Local PipeConfig entry in config.xml
//
void XMLConfigWriter::localPipe(const QString& host)
{
   //find LOCAL for host and the COVER-Child and find/create PipeConfig
   //(only for 1 graphic card)
   QDomElement pipeConfig = findCreateLocalHostCoverSpecElem(host, "PipeConfig");
   QDomElement pipeElem = domDocVrcw.createElement("Pipe");
   pipeElem.setAttribute("display", ":0.0");
   pipeElem.setAttribute("name", 0);
   pipeElem.setAttribute("screen", 0);
   pipeElem.setAttribute("pipe", 0);
   pipeConfig.appendChild(pipeElem);
}

//generate Local WindowConfig entry in config.xml
//
void XMLConfigWriter::localWindow(const QString& host, const QString& comment,
      const int& width, const int& height, const bool& stereo,
      const int& left, const int& bottom, const QString& decoration,
      const int& window, const int& pipeIndex, const int& name)
{
   //find LOCAL for host and the COVER-Child and find/create WindowConfig
   QDomElement windowConfig = findCreateLocalHostCoverSpecElem(host,
         "WindowConfig");
   QDomElement windowElem = domDocVrcw.createElement("Window");
   windowElem.setAttribute("comment", comment);//QString, blablabla
   windowElem.setAttribute("window", window);//Index, normal int, 0
   windowElem.setAttribute("pipeIndex", pipeIndex);//Index, normal int, 0
   windowElem.setAttribute("width", width);//int, aus ProjectionDimXXX
   windowElem.setAttribute("height", height);//int, aus ProjectionDimXXX
   if (stereo)
   {
      windowElem.setAttribute("stereo", "true");//QString true fuer AktivStereo
   }
   windowElem.setAttribute("left", left);//Offset auf dem X-Server,
                                         //z.B. bei Twinview
   windowElem.setAttribute("bottom", bottom);//Offset auf dem X-Server,
                                             //z.B. bei Twinview
   windowElem.setAttribute("name", name);//int, normal 0
   windowElem.setAttribute("decoration", decoration);//QString
   windowConfig.appendChild(windowElem);
}

//generate Local ChannelConfig entry in config.xml
//
void XMLConfigWriter::localChannel(const QString& host, const QString& comment,
      const QString& stereoMode, const int& channel, const int& windowIndex,
      const int& viewportIndex, const int& name)
{
   //find LOCAL for host and the COVER-Child and find/create ChannelConfig
   QDomElement channelConfig = findCreateLocalHostCoverSpecElem(host,
         "ChannelConfig");
   QDomElement channelElem = domDocVrcw.createElement("Channel");
   channelElem.setAttribute("comment", comment);//QString, blablabla
   channelElem.setAttribute("channel", channel);//int, normal 0
   channelElem.setAttribute("stereoMode", stereoMode);//QString, abh. von Eye
   channelElem.setAttribute("windowIndex", windowIndex);//int, normal 0
   channelElem.setAttribute("viewportIndex", viewportIndex);//QString,
                                                            //abh. von Eye
   channelElem.setAttribute("name", name);//int, normal 0
   channelConfig.appendChild(channelElem);
}

//generate Local ViewportConfig entry in config.xml
//
void XMLConfigWriter::localViewport(const QString& host, const int& width,
      const int& height, const int& left, const int& bottom,
      const int& windowIndex, const int& name)
{
   //find LOCAL for host and the COVER-Child and find/create ViewportConfig
   QDomElement viewportConfig = findCreateLocalHostCoverSpecElem(host,
         "ViewportConfig");
   QDomElement viewportElem = domDocVrcw.createElement("Viewport");
   viewportElem.setAttribute("windowIndex", windowIndex);//int, normal 0
   viewportElem.setAttribute("left", left);//int, PW, CAVE abh. von screen/wall
                                           //size and resolution
   viewportElem.setAttribute("bottom", bottom);//int. PW, CAVE abh. von screen/
                                               //wall size and resolution
   viewportElem.setAttribute("width", width);//int, aus ProjectionDimXXX
   viewportElem.setAttribute("height", height);//int, aus ProjectionDimXXX
   viewportElem.setAttribute("name", name);//int, normal 0
   viewportConfig.appendChild(viewportElem);

}

//generate Local ScreenConfig entry in config.xml
//
void XMLConfigWriter::localScreen(const QString& host, const QString& comment,
      const int& width, const int& height,
      const int& originX, const int& originY, const int& originZ,
      const double& h, const double& p, const double& r,
      const int& name, const int& screen)
{
   //find LOCAL for host and the COVER-Child and find/create ScreenConfig
   QDomElement screenConfig = findCreateLocalHostCoverSpecElem(host,
         "ScreenConfig");
   QDomElement screenElem = domDocVrcw.createElement("Screen");
   screenElem.setAttribute("comment", comment);//QString, blablabla
   screenElem.setAttribute("width", width);//int, aus ProjectionDimXXX
   screenElem.setAttribute("height", height);//int, aus ProjectionDimXXX
   screenElem.setAttribute("originX", originX);//PW = 0, CAVE abh. von Wall
   screenElem.setAttribute("originY", originY);//PW = 0, CAVE abh. von Wall
   screenElem.setAttribute("originZ", originZ);//PW = 0, CAVE abh. von Wall
   screenElem.setAttribute("h", QString::number(h, 'f', 1));//PW = 0.0, CAVE
                                                            //abh. von Wall
   screenElem.setAttribute("p", QString::number(p, 'f', 1));//PW = 0.0, CAVE
                                                            //abh. von Wall
   screenElem.setAttribute("r", QString::number(r, 'f', 1));//PW = 0.0, CAVE
                                                            //abh. von Wall
   screenElem.setAttribute("name", name);//int, normal 0
   screenElem.setAttribute("screen", screen);//Index, normal int, 0
   screenConfig.appendChild(screenElem);
}

//generate Input basic entry in config.xml
//
void XMLConfigWriter::inputBasic()
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   QDomElement mouseNav = findCreateSpecElem(input, "MouseNav");
   mouseNav.setAttribute("value", "true");
}

//generate Local Input Device TrackingSys entry with name tSysName
//in config.xml
//
void XMLConfigWriter::inputTrackSysDev(const QString& tSysName,
      const QString& drv, sensTrackSysDim*& trackSysDim)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //Device
   QDomElement device = findCreateSpecElem(input, "Device");

   //Tracking System
   QDomElement trackSys = findCreateSpecElem(device, tSysName);

   trackSys.setAttribute("driver", drv);

   //Offset and Orientation
   QDomElement offset = domDocVrcw.createElement("Offset");
   offset.setAttribute("x", trackSysDim->x);
   offset.setAttribute("y", trackSysDim->y);
   offset.setAttribute("z", trackSysDim->z);
   trackSys.appendChild(offset);
   QDomElement orient = domDocVrcw.createElement("Orientation");
   orient.setAttribute("h", QString::number(double(trackSysDim->h), 'f', 1));
   orient.setAttribute("p", QString::number(double(trackSysDim->p), 'f', 1));
   orient.setAttribute("r", QString::number(double(trackSysDim->r), 'f', 1));
   trackSys.appendChild(orient);
}

//generate Local Input Device ButtonDev with name bDevName entry in config.xml
//
void XMLConfigWriter::inputButtonDev(const QString& bDevName,
      const QString& drv, const QString& dev)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //Device
   QDomElement device = findCreateSpecElem(input, "Device");

   //Button Device
   QDomElement btnDev = findCreateSpecElem(device, bDevName);

   btnDev.setAttribute("driver", drv);

   if (!dev.isEmpty())
   {
      btnDev.setAttribute("device", dev);
   }
}

//adds attribute to Element with name elem in Local input Device section
//
void XMLConfigWriter::addAttrToElemInDev(const QString& elem,
      const QString& attrName, const QString& attrVal)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //Device
   QDomElement device = findCreateSpecElem(input, "Device");

   //Element
   QDomElement element = findCreateSpecElem(device, elem);

   element.setAttribute(attrName, attrVal);
}



//generate Local input Body entry with name bName in config.xml
//
void XMLConfigWriter::inputBody(const QString& bName,
      sensTrackSysDim*& hhSensDim, const QString& dev,
      const int& bIndex)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //body
   QDomElement body = findCreateSpecElem(input, "Body");

   //hand or head sensors
   QDomElement hhSens = findCreateSpecElem(body, bName);

   //device and bodyIndex
   if (!dev.isEmpty())
   {
      hhSens.setAttribute("device", dev);
   }
   if (bIndex > -1) //-1 is initial value of bIndex in function
   {
      hhSens.setAttribute("bodyIndex", bIndex);
   }

   //Offset and Orientation
   QDomElement offset = domDocVrcw.createElement("Offset");
   offset.setAttribute("x", hhSensDim->x);
   offset.setAttribute("y", hhSensDim->y);
   offset.setAttribute("z", hhSensDim->z);
   hhSens.appendChild(offset);
   QDomElement orient = domDocVrcw.createElement("Orientation");
   orient.setAttribute("h", QString::number(double(hhSensDim->h), 'f', 1));
   orient.setAttribute("p", QString::number(double(hhSensDim->p), 'f', 1));
   orient.setAttribute("r", QString::number(double(hhSensDim->r), 'f', 1));
   hhSens.appendChild(orient);
}


//generate Local input Buttons entry with name btnsName in config.xml
//
void XMLConfigWriter::inputButtons(const QString& btnsName,
      const QString& btnDev)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //Buttons
   QDomElement buttons = findCreateSpecElem(input, "Buttons");

   //ButtonDevice
   QDomElement buttonDev = findCreateSpecElem(buttons, btnsName);
   buttonDev.setAttribute("device", btnDev);

   QString action = "ACTION_BUTTON";
   QString xform = "XFORM_BUTTON";
   QString menu = "MENU_BUTTON";
   QString drive = "DRIVE_BUTTON";
   QVector<QString> mapButton;

   //ButtonMap
   mapButton.append(action);
   mapButton.append(xform);
   mapButton.append(menu);
   mapButton.append(drive);

   for (int i = 0; i < mapButton.size(); ++i)
   {
      QDomElement buttonMap = domDocVrcw.createElement("Map");
      buttonMap.setAttribute("name", QString::number(i));
      buttonMap.setAttribute("value", mapButton[i]);
      buttonDev.appendChild(buttonMap);
   }
}

//generate Local input Persons entry in config.xml
void XMLConfigWriter::inputPersons(const QString& pName, const QString& hand,
      const QString& head, const QString& button)
{
   //find or create Input as a child of GlobalCover
   QDomElement input = findCreateSpecElem(gCover, "Input");

   //Persons
   QDomElement persons = findCreateSpecElem(input, "Persons");

   //Person
   QDomElement pers = domDocVrcw.createElement("Person");
   pers.setAttribute("name", pName);
   pers.setAttribute("hand", hand);
   pers.setAttribute("head", head);
   pers.setAttribute("buttons", button);
   persons.appendChild(pers);
}


//Output of domDocVrcw
//
QString XMLConfigWriter::strOutVrcw(const int& indent, QTextCodec* codec) const
{
   QString xmlStrOut;
   QTextStream out(&xmlStrOut, QIODevice::WriteOnly);
   out.setCodec(codec);
   domDocVrcw.save(out, indent);

   return xmlStrOut;
}

//generate domDocUser for configUser.xml
//
void XMLConfigWriter::generateConfigUser(const QString& coConfNameVrcw)
{
   domDocUser.clear();

   QDomElement coConfUser = domDocUser.createElement("COCONFIG");
   coConfUser.setAttribute("version", "1");
   domDocUser.appendChild(coConfUser);

   QDomComment commentOrder1 = domDocUser.createComment("=============="
         "============== ATTENTION: ============================");
   coConfUser.appendChild(commentOrder1);
   QDomComment commentOrder2 = domDocUser.createComment("= ORDER OF "
         "SETTINGS IS IMPORTANT, SETTINGS WILL NOT BE OVERWRITTEN =");
      coConfUser.appendChild(commentOrder2);

   QDomElement includeDef = domDocUser.createElement("INCLUDE");
   includeDef.setAttribute("global","1");
   includeDef.setAttribute("configname", "defaultConfig");
   coConfUser.appendChild(includeDef);
   QDomText textDef = domDocUser.createTextNode("config.xml");
   includeDef.appendChild(textDef);

   QDomElement includeVRCW = domDocUser.createElement("INCLUDE");
   includeVRCW.setAttribute("global","1");
   includeVRCW.setAttribute("configname", "VRCW-generated");
   coConfUser.appendChild(includeVRCW);
   QDomText textVRCW = domDocUser.createTextNode(coConfNameVrcw);
   includeVRCW.appendChild(textVRCW);

   QDomElement global = domDocUser.createElement("GLOBAL");
   coConfUser.appendChild(global);

   QDomComment commentUser = domDocUser.createComment("========== "
         "Put your GLOBAL settings after this comment ==========");
   global.appendChild(commentUser);

   QDomComment commentLocal = domDocUser.createComment("= For LOCAL "
         "settings start here with: <LOCAL host=\"your-hostname\"> =");
   coConfUser.appendChild(commentLocal);

   QDomNode xmlNodeUser = domDocUser.createProcessingInstruction("xml",
         "version=\"1.0\"");
   domDocUser.insertBefore(xmlNodeUser, domDocUser.firstChild());
}

//Output of domDocUser
//
QString XMLConfigWriter::strOutUser(const int& indent, QTextCodec* codec) const
{
   QString xmlStrOut;
   QTextStream out(&xmlStrOut, QIODevice::WriteOnly);
   out.setCodec(codec);
   domDocUser.save(out, indent);

   return xmlStrOut;
}


/*****
 * private functions
 *****/

//find or create QDomElement LOCAL with attribute host
//
QDomElement XMLConfigWriter::findCreateLocalHostElem(const QString& host)
{
   QDomNodeList localNodes = domDocVrcw.elementsByTagName("LOCAL");
   QDomElement localHostElem;
   bool foundElem = false;

   if (!localNodes.isEmpty())
   {
      for (int i = 0; i < localNodes.size(); ++i)
      {
         QDomElement local = localNodes.item(i).toElement();
         QString localAttrHostValue = local.attribute("host");

         if (localAttrHostValue == host)
         {
            localHostElem = local;
            foundElem = true;
         }
      }
   }

   if (!foundElem)
   {
      //LOCAL-Element erzeugen
      localHostElem = domDocVrcw.createElement("LOCAL");
      localHostElem.setAttribute("host", host);
      coConf.appendChild(localHostElem);
   }

   return localHostElem;
}

//find or create special QDomElement specElemStr as child of a
//specified QDomElement element
//
QDomElement XMLConfigWriter::findCreateSpecElem(QDomElement& element,
      const QString& specElemStr, const char& insParam)
{
   QDomElement specElemTmp = element.firstChildElement(specElemStr);
   QDomElement specElem;

   if (specElemTmp.isNull())
   {
      specElem = domDocVrcw.createElement(specElemStr);

      switch (insParam)
      {
         case 'p':
         {
            element.insertBefore(specElem, element.firstChild());
            break;
         }
         case 'a': //a should be the default
         default:
         {
            element.appendChild(specElem);
            break;
         }
      }
   }
   else
   {
      specElem = specElemTmp;
   }

   return specElem;
}

//find or create special QDomElement specElemStr as child of
//a COVER QDomElement of a Local QDomElement for specified host
//
QDomElement XMLConfigWriter::findCreateLocalHostCoverSpecElem(
      const QString& host, const QString& specElemStr, const char& insParam)
{
   //find LOCAL for host and the COVER-Child
   QDomElement localHostElem = findCreateLocalHostElem(host);
   QDomElement localHostCoverElem = findCreateSpecElem(localHostElem, "COVER");

   //specElemStr Config
   return findCreateSpecElem(localHostCoverElem, specElemStr, insParam);
}
