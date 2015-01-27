/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** configfileio.cpp
 ** 2004-02-04, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include <qfile.h>
#include <qtextstream.h>
#include "configfileio.h"
#include "covergeneral.h"
#include "host.h"
#include "projectionarea.h"

const QString windowConfigComment = "#-- WinIndex\tWinName\tSoftPipeNo\tOrigin[pixel]\tSize[pixel]";
const QString pipeConfigComment = "#-- PipeIndex\tHardPipe\tDisplay";
const QString channelConfigComment = "#-- ChannelIndex\tChannelName\tWindowNo\tVPLeft\tVPBottom\tVPRight\tVPTop";
const QString screenConfigComment = "#-- ScreenIndex\tScreenName\tScreenSize[mm]\tScreenCenter[mm]\tScreenOrient hpr [degrees]";

/*------------------------------------------------------------------------------
 ** ConfigFileIO::ConfigFileIO()
 **     constructor, sets member variables to default values.
-------------------------------------------------------------------------------*/
ConfigFileIO::ConfigFileIO()
{
    hostMap = 0;
    projMap = 0;
    fileName = QString();
    genSets = 0;
    tracking = 0;
}

/*------------------------------------------------------------------------------
 ** HostMap* ConfigFileIO::getHostMap()
 **     returns actual hostMap
-------------------------------------------------------------------------------*/
HostMap *ConfigFileIO::getHostMap()
{
    return hostMap;
}

/*------------------------------------------------------------------------------
 ** ProjectionAraMap* ConfigFileIO::getProjMap()
 **     returns actual projMap
-------------------------------------------------------------------------------*/
ProjectionAreaMap *ConfigFileIO::getProjMap()
{
    return projMap;
}

/*------------------------------------------------------------------------------
 ** CoverGeneral* ConfigFileIO::getGeneralSettings()
 **     returns actual general settings
-------------------------------------------------------------------------------*/
CoverGeneral *ConfigFileIO::getGeneralSettings()
{
    return genSets;
}

/*------------------------------------------------------------------------------
 ** Tracking* ConfigFileIO::getTracking()
 **     returns actual tracking
-------------------------------------------------------------------------------*/
Tracking *ConfigFileIO::getTracking()
{
    return tracking;
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::setHostMap(HostMap* hm)
 **     sets new host map
-------------------------------------------------------------------------------*/
void ConfigFileIO::setHostMap(HostMap *hm)
{
    hostMap = hm;
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::setProjMap(ProjMap* pm)
 **     sets new proj map
-------------------------------------------------------------------------------*/
void ConfigFileIO::setProjMap(ProjectionAreaMap *pm)
{
    projMap = pm;
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::setGeneralSettings(CoverGeneral* gs)
 **     sets new general settings
-------------------------------------------------------------------------------*/
void ConfigFileIO::setGeneralSettings(CoverGeneral *gs)
{
    genSets = gs;
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::setTracking(Tracking* t)
 **     sets new general settings
-------------------------------------------------------------------------------*/
void ConfigFileIO::setTracking(Tracking *t)
{
    tracking = t;
}

/*------------------------------------------------------------------------------
 ** bool ConfigFileIO::saveConfigFile(QString name)
 **     saves configFile with filename name and uses the information
 **     which was set with the "set"-functions.
-------------------------------------------------------------------------------*/
bool ConfigFileIO::saveConfigFile(QString name)
{
    QString sectionName;
    QStringList hostList;
    QStringList valueList;
    QStringList screenValueList;
    QString line;

    Host h;
    Pipe pipe;
    Window win;
    Channel ch;

    fileName = name;

    QFile file(fileName);
    if (file.open(IO_WriteOnly)) // opens file and deletes old content
    {
        file.close();

        // write head line
        writeCommentLine("#####################################################################################");
        writeCommentLine("# +----------------------------------------------------------------------------------");
        writeCommentLine("# | COVER Configuration Tool");
        writeCommentLine("# +----------------------------------------------------------------------------------");
        writeCommentLine("\n");

        // check if projMap, general settings and hostMap was set
        if ((projMap != 0) && (genSets->getStereoMode() != QString()) && (hostMap != 0))
        {

            // -------------------- M U L T I   P C    C O N F I G -------------------
            // begin with MultiPC Config
            if (hostMap->count() > 1)
                writeMultiPCConfig();

            // -------------------- C O V E R   C O N F I G  G E N E R A L ---------------
            writeCoverConfigGeneral();

            // iterate over all hosts
            HostMap::Iterator hostIt;
            for (hostIt = hostMap->begin(); hostIt != hostMap->end(); ++hostIt)
            {
                h = hostIt.data();
                //pipeMap = h.getPipeMap();

                if ((!h.isControlHost()) && (h.getNumChannels() > 0))
                {

                    writeCommentLine("# +----------------------------------------------------------------------------------");
                    writeCommentLine(QString("# | Host: ").append(h.getName()));
                    writeCommentLine("# +----------------------------------------------------------------------------------");
                    writeCommentLine("\n");

                    // -------------------- C O V E R    C O N F I G   H O S T------------------
                    writeCoverConfigHost(&h);

                    // ---------------------------- P I P E  C O N F I G ----------------------
                    writePipeConfig(&h);

                    // ---------------------------- W I N D O W   C O N F I G -------------------

                    writeWindowConfig(&h);

                    // ---------------------------- C H A N N E L   C O N F I G ---------------
                    // ----------------------   A N D   S C R E E N   C O N F I G   -----------

                    writeChannelAndScreenConfig(&h);
                }
            }
            // ---------------------------- T R A C K E R   C O N F I G -------------------
            writeTrackerConfig();
        }
        return true;
    }
    else
        return false;
}

//void loadConfigFile(QSring fileName, HostMap* hm, CoverGeneral* gc){}

//private:

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeMultiPCConfig()
 **     writes the multiPCConfig section to the config file.
 **   Parameters:
 **     NONE
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeMultiPCConfig()
{
    QString sectionName;
    QString masterHost;
    QString masterInterface;
    QStringList hostList;
    QStringList valueList;
    QStringList slaveList;
    QString line;
    int numSlaves = 0;

    Host h;
    sectionName = "MultiPC";
    valueList = "{";

    writeCommentLine("#------------------------------------------------------------------------------------");

    line = "\t";
    line.append("SyncMode\t");

    // syncMode
    if (genSets->getSyncMode() == SyncModeType(TCP))
        line.append("TCP");
    else if (genSets->getSyncMode() == SyncModeType(SERIAL))
        line.append("SERIAL");
    line.append("\n");

    // syncProcess
    line.append("\tSyncMode\t");
    if (genSets->getSyncProcess() == SyncProcessType(APP))
        line.append("APP");
    else if (genSets->getSyncProcess() == SyncProcessType(DRAW))
        line.append("DRAW");
    line.append("\n");
    line.append("\tSerialDevice\t").append(genSets->getSerialDevice());
    valueList.append(line);

    // iterate over all hosts to get host list, master host, numSlaves,...
    HostMap::Iterator hostIt;
    for (hostIt = hostMap->begin(); hostIt != hostMap->end(); ++hostIt)
    {
        line = "";
        h = hostIt.data();
        hostList.append(h.getName());
        if (h.isMasterHost())
        {
            masterHost.append("\tMaster\t\t").append(h.getName());
            masterInterface.append("\tMasterInterface\t").append(h.getMasterInterface());
        }
        else if (h.isControlHost())
        {
        }
        else // slave
        {
            line.append("\tHost").append(QString().setNum(numSlaves)).append("\t\t");
            line.append(h.getName());
            numSlaves++;
            slaveList.append(line);
        }
    }
    valueList.append(masterHost);
    valueList.append(masterInterface);
    valueList += slaveList;
    line = "\t# command ---> not implemented yet!\n";
    line.append("\tnumSlaves\t").append(QString().setNum(numSlaves));
    valueList.append(line);
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeCoverConfig()
 **     writes the COVERConfig section for give host h to the config file.
 **   Parameters:
 **     Host* h
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeCoverConfigGeneral()
{
    QString sectionName = "COVERConfig";
    QStringList hostList;
    QStringList valueList;
    QString line;
    Host h;

    // iterate over all hosts to get host list
    HostMap::Iterator hostIt;
    for (hostIt = hostMap->begin(); hostIt != hostMap->end(); ++hostIt)
    {
        h = hostIt.data();
        if (!h.isControlHost())
            hostList.append(h.getName());
    }

    valueList = "{";

    line = "\t";
    line.append("VIEWER_POSITION\t\t");
    line.append(QString().setNum(genSets->getViewerPosX()).append(" "));
    line.append(QString().setNum(genSets->getViewerPosY()).append(" "));
    line.append(QString().setNum(genSets->getViewerPosZ()).append("\n"));
    line.append("\tMENU_POSITION\t");
    line.append("\t").append(QString().setNum(genSets->getMenuPosX()).append(" "));
    line.append(QString().setNum(genSets->getMenuPosY()).append(" "));
    line.append(QString().setNum(genSets->getMenuPosZ()).append("\n"));
    line.append("\tMENU_ORIENTATION");
    line.append("\t").append(QString().setNum(genSets->getMenuOrient_h()).append(" "));
    line.append(QString().setNum(genSets->getMenuOrient_p()).append(" "));
    line.append(QString().setNum(genSets->getMenuOrient_r()).append("\n"));
    line.append("\tMENU_SIZE");
    line.append("\t\t").append(QString().setNum(genSets->getMenuSize()).append("\n"));
    line.append("\tSCENESIZE");
    line.append("\t\t").append(QString().setNum(genSets->getSceneSize()).append("\n"));
    line.append("\tfloorHeight");
    line.append("\t\t").append(QString().setNum(genSets->getFloorHeight()).append("\n"));
    line.append("\tstepSize");
    line.append("\t\t").append(QString().setNum(genSets->getStepSize()).append(""));

    valueList.append(line);
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeTrackerConfig()
 **     writes the TrackerConfig section.
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeTrackerConfig()
{
    QString sectionName = "TrackerConfig";
    QStringList hostList;
    QStringList valueList;
    QString line;
    Host h;

    // iterate over all hosts to get master host
    HostMap::Iterator hostIt;
    for (hostIt = hostMap->begin(); hostIt != hostMap->end(); ++hostIt)
    {
        h = hostIt.data();
        if (h.isMasterHost())
        {
            hostList.append(h.getName());
            break;
        }
    }

    valueList = "{";

    line = "\t";
    line.append("# number of connected sensors, id of head sensor, id of hand sensor\n");
    ;
    line.append("\tNUM_SENSORS\t");
    line.append(QString().setNum(tracking->getNoSensors()).append("\n"));
    line.append("\tHEAD_ADDR\t");
    line.append(QString().setNum(tracking->getAdrHeadSensor()).append("\n"));
    line.append("\tHAND_ADDR\t");
    line.append(QString().setNum(tracking->getAdrHandSensor()).append("\n\n"));

    line.append("\t# offset and orientation of transmitter in the world coordinate system.\n");
    line.append("\tTRANSMITTER_OFFSET");
    line.append("\t").append(QString().setNum(tracking->getTransmitterOffsetX()).append(" "));
    line.append(QString().setNum(tracking->getTransmitterOffsetY()).append(" "));
    line.append(QString().setNum(tracking->getTransmitterOffsetZ()).append("\n"));
    line.append("\tTRANSMITTER_ORIENTATION");
    line.append("\t").append(QString().setNum(tracking->getTransmitterOrientH()).append(" "));
    line.append(QString().setNum(tracking->getTransmitterOrientP()).append(" "));
    line.append(QString().setNum(tracking->getTransmitterOrientR()).append("\n\n"));

    line.append("\t# offset and orientation of viewer in the sensor coordinate system.\n");
    line.append("\tHEADSENSOR_OFFSET");
    line.append("\t").append(QString().setNum(tracking->getHeadSensorOffsetX()).append(" "));
    line.append(QString().setNum(tracking->getHeadSensorOffsetY()).append(" "));
    line.append(QString().setNum(tracking->getHeadSensorOffsetZ()).append("\n"));
    line.append("\tHEADSENSOR_ORIENTATION");
    line.append("\t").append(QString().setNum(tracking->getHeadSensorOrientH()).append(" "));
    line.append(QString().setNum(tracking->getHeadSensorOrientP()).append(" "));
    line.append(QString().setNum(tracking->getHeadSensorOrientR()).append("\n\n"));

    line.append("\t# offset and orientation of hand in the stylus (pen) coordinate system.\n");
    line.append("\tHANDSENSOR_OFFSET");
    line.append("\t").append(QString().setNum(tracking->getHandSensorOffsetX()).append(" "));
    line.append(QString().setNum(tracking->getHandSensorOffsetY()).append(" "));
    line.append(QString().setNum(tracking->getHandSensorOffsetZ()).append("\n"));
    line.append("\tHANDSENSOR_ORIENTATION");
    line.append("\t").append(QString().setNum(tracking->getHandSensorOrientH()).append(" "));
    line.append(QString().setNum(tracking->getHandSensorOrientP()).append(" "));
    line.append(QString().setNum(tracking->getHandSensorOrientR()).append("\n\n"));

    line.append("\t# linear magnetic field correction in [cm]\n");
    ;
    line.append("\tLINEAR_MAGNETIC_FIELD_CORRECTION");
    line.append("\t").append(QString().setNum(tracking->getLinearMagneticFieldCorrectionX()).append(" "));
    line.append(QString().setNum(tracking->getLinearMagneticFieldCorrectionY()).append(" "));
    line.append(QString().setNum(tracking->getLinearMagneticFieldCorrectionZ()).append("\n"));
    line.append("\tINTERPOLATION_FILE");
    line.append("\t").append(tracking->getInterpolationFile().append("\n\n"));

    line.append("\t# debugging options: \n");
    line.append("\t# DEBUG_TRACKING (RAW: untransformed, APP: in coord. system of application)\n");
    line.append("\t# DEBUG_BUTTONS (TRUE, FALSE)\n");
    line.append("\t# DEBUG STATION (id of sensor)\n");

    line.append("\tDEBUG_TRACKING");
    line.append("\t").append(tracking->getDebugTrackingString().append("\n"));
    line.append("\tDEBUG_BUTTONS");
    line.append("\t").append(tracking->getDebugButtonsString().append("\n"));
    line.append("\tDEBUG_STATION");
    line.append("\t").append(QString().setNum(tracking->getDebugStation()).append(""));

    valueList.append(line);
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeCoverConfigHost()
 **     writes the COVERConfig section for given host h to the config file.
 **   PRE: this function is only called for non-control-hosts.
 **   Parameters:
 **     Host* h
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeCoverConfigHost(Host *h)
{
    QString sectionName = "COVERConfig";
    QStringList hostList = QStringList(h->getName());
    ;
    QStringList valueList;
    QString line;

    valueList = "{";

    line = "\t";
    line.append("NUM_PIPES");
    line.append("\t").append(QString().setNum(h->getNumPipes())).append("\n");
    line.append("\tNUM_WINDOWS");
    line.append("\t").append(QString().setNum(h->getNumWindows())).append("\n");
    line.append("\tNUM_SCREENS");
    line.append("\t").append(QString().setNum(h->getNumChannels())).append("\n");

    // MonoView
    if (genSets->getStereoMode() == "passive") // MONO VIEW NEEDED
    {
        line.append("\tMONO_VIEW");
        line.append("\t").append(h->getMonoView()).append("\n");
    }

    QString trackingSystem;
    // tracking sytem... note, that h is not control host by PRE-condition.
    line.append("\tTRACKING_SYSTEM");
    if (h->isMasterHost())
    {
        switch (h->getTrackingSystem())
        {
        case POLHEMUS:
            trackingSystem = "POLHEMUS";
            break;
        case MOTIONSTAR:
            trackingSystem = "MOTIONSTAR";
            break;
        case FOB:
            trackingSystem = "FOB";
            break;
        case DTRACK:
            trackingSystem = "DTRACK";
            break;
        case VRC:
            trackingSystem = "VRC";
            break;
        case CAVELIB:
            trackingSystem = "CAVELIB";
            break;
        case SPACEBALL:
            trackingSystem = "SPACEBALL";
            break;
        case SPACEPOINTER:
            trackingSystem = "SPACEPOINTER";
            break;
        case MOUSE:
            trackingSystem = "MOUSE";
            break;
        case NONE:
            trackingSystem = "NONE";
            break;
        }
    }
    else
        trackingSystem = "NONE";

    line.append("\t").append(trackingSystem);

    valueList.append(line);
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writePipeConfig(QString host,
 **		                   PipeMap pipeMap)
 **     writes the pipeConfig section to the config file.
 **   Parameters:
 **     host:            the host for which we write the section
-------------------------------------------------------------------------------*/
void ConfigFileIO::writePipeConfig(Host *h)
{
    QString sectionName;
    QStringList hostList = QStringList(h->getName());
    QStringList valueList;
    QString line;

    PipeMap *pipeMap = h->getPipeMap();
    PipeMap::Iterator pipeIt;
    Pipe pipe;

    sectionName = "PipeConfig";
    valueList = "{";
    valueList.append(pipeConfigComment);

    // iterate over all pipes and get pipe data
    for (pipeIt = pipeMap->begin(); pipeIt != pipeMap->end(); ++pipeIt)
    {

        pipe = pipeIt.data();
        line = "\t";
        line.append(QString().setNum(pipe.getIndex()).append("\t"));
        line.append(QString().setNum(pipe.getHardPipe()).append("\t\t"));
        line.append(pipe.getDisplay());
        valueList.append(line);
    }
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeWindowConfig(QString host,
 **		                   PipeMap pipeMap)
 **     writes the windowConfig section to the config file.
 **   Parameters:
 **     host:            the host for which we write the section
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeWindowConfig(Host *h)
{
    QString sectionName;
    QStringList hostList;
    QStringList valueList;
    QString line;

    WindowMap *winMap;
    PipeMap *pipeMap = h->getPipeMap();
    PipeMap::Iterator pipeIt;
    WindowMap::Iterator winIt;
    Pipe pipe;
    Window win;

    int winOffset = 0; // used for numerating the windows consecutively

    sectionName = "WindowConfig";
    hostList = QStringList(h->getName());
    valueList = "{";
    valueList.append(windowConfigComment);
    winOffset = 0; // used for numerating the windows consecutively

    //  iterate over all pipes
    for (pipeIt = pipeMap->begin(); pipeIt != pipeMap->end(); ++pipeIt)
    {

        pipe = pipeIt.data();
        winMap = pipe.getWindowMap();

        // iterate over all windows of pipe
        for (winIt = winMap->begin(); winIt != winMap->end(); ++winIt)
        {
            win = winIt.data();
            line = "\t";
            line.append(QString().setNum(win.getIndex() + winOffset).append("\t"));

            QString winName = createWinName(&win);
            line.append(winName.append("\t"));

            line.append(QString().setNum(pipe.getIndex()).append("\t\t"));
            line.append(QString().setNum(win.getOriginX()).append(" "));
            line.append(QString().setNum(win.getOriginY()).append("\t\t"));
            line.append(QString().setNum(win.getWidth()).append(" "));
            line.append(QString().setNum(win.getHeight()));
            valueList.append(line);
        }
        winOffset += pipe.getNumWindows();
    }
    valueList.append("}");
    writeSection(sectionName, hostList, valueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeChannelAndScreenConfig(QString host,
 **     writes the channel and screen config section to the config file.
 **   Parameters:
 **     host:            the host for which we write the section
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeChannelAndScreenConfig(Host *h)
{
    QString sectionName;
    QStringList hostList;
    QStringList valueList;
    QStringList screenValueList;
    QString line;

    PipeMap *pipeMap = h->getPipeMap();
    WindowMap *winMap;
    ChannelMap *chMap;
    PipeMap::Iterator pipeIt;
    WindowMap::Iterator winIt;
    ChannelMap::Iterator chIt;
    Pipe pipe;
    Window win;
    Channel ch;
    ProjectionArea *pArea;

    int winOffset = 0; // used for numerating the windows consecutively
    int channelOffset = 0; // used for numerating the channels continuously

    sectionName = "ChannelConfig";
    hostList = QStringList(h->getName());
    valueList = "{";
    valueList.append(channelConfigComment);
    screenValueList = "{";
    screenValueList.append(screenConfigComment);
    channelOffset = 0; // used for numerating the channels continuously
    winOffset = 0; // used for numerating the channels continuously

    //  iterate over all pipes
    for (pipeIt = pipeMap->begin(); pipeIt != pipeMap->end(); ++pipeIt)
    {
        pipe = pipeIt.data();
        winMap = pipe.getWindowMap();

        // iterate over all windows of pipe
        for (winIt = winMap->begin(); winIt != winMap->end(); ++winIt)
        {
            win = winIt.data();
            chMap = win.getChannelMap();

            // iterate over all channels of window
            for (chIt = chMap->begin(); chIt != chMap->end(); ++chIt)
            {
                ch = chIt.data();
                line = "\t";
                line.append(QString().setNum(ch.getIndex() + channelOffset).append("\t\t"));

                QString chName = createChannelName(&ch);
                line.append(chName.append("\t\t"));

                line.append(QString().setNum(win.getIndex() + winOffset).append("\t\t"));
                line.append(QString().setNum(ch.getLeft()).append("\t"));
                line.append(QString().setNum(ch.getBottom()).append("\t\t"));
                line.append(QString().setNum(ch.getRight()).append("\t"));
                line.append(QString().setNum(ch.getTop()));
                valueList.append(line);

                if (ch.getProjectionArea() != 0)
                {
                    pArea = ch.getProjectionArea();
                    line = "\t";
                    line.append(QString().setNum(ch.getIndex() + channelOffset).append("\t"));
                    line.append(pArea->getName().append("\t\t"));
                    line.append(QString().setNum(pArea->getWidth()).append(" "));
                    line.append(QString().setNum(pArea->getHeight()).append("\t\t"));
                    line.append(QString().setNum(pArea->getOriginX()).append(" "));
                    line.append(QString().setNum(pArea->getOriginY()).append(" "));
                    line.append(QString().setNum(pArea->getOriginZ()).append("\t\t\t"));
                    line.append(QString().setNum(pArea->getRotation_h()).append(" "));
                    line.append(QString().setNum(pArea->getRotation_p()).append(" "));
                    line.append(QString().setNum(pArea->getRotation_r()));
                }
                screenValueList.append(line);
            }
            channelOffset += win.getNumChannels();
        }
        winOffset += pipe.getNumWindows();
    }
    valueList.append("}");
    screenValueList.append("}");
    writeSection(sectionName, hostList, valueList);
    writeSection("ScreenConfig", hostList, screenValueList);
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeSection(QString sectionName,
 **       		           QStringList hostList,
 **		                   QStringList valueList)
 **     writes a section to the config file.
 **   Parameters:
 **     sectionName:         name of the section
 **     hostList:            the host for which we write the section
 **     valueList:           values to be written, organized in lines
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeSection(QString sectionName,
                                QStringList hostList,
                                QStringList valueList)
{
    QFile file(fileName);
    if (file.open(IO_ReadWrite))
    {
        QTextStream stream(&file);

        // first read until we're at the end of the stream
        while (!stream.atEnd())
        {
            stream.readLine();
        }

        // append host names to the section name
        sectionName.append(": ");
        for (unsigned int i = 0; i < hostList.count(); i++)
        {
            sectionName.append(hostList[i]).append(" ");
        }

        // crate a new QStringList with all lines
        QStringList lines = QStringList();
#if defined(CO_sgi64) || defined(CO_sgin32) || defined(IRIX)
        lines = "";
#endif
        lines.append(sectionName);
        lines += valueList;
        lines.append("\n");

        for (QStringList::Iterator it = lines.begin(); it != lines.end(); ++it)
            stream << *it << "\n";
        file.close();
    }
}

/*------------------------------------------------------------------------------
 ** void ConfigFileIO::writeSection(QString commentString)
 **     writes a comment line to the config file.
 **   Parameters:
 **     commentString:       the comment to be written.
-------------------------------------------------------------------------------*/
void ConfigFileIO::writeCommentLine(QString commentString)
{

    QFile file(fileName);
    if (file.open(IO_ReadWrite))
    {
        QTextStream stream(&file);

        // first read until we're at the end of the stream
        while (!stream.atEnd())
        {
            stream.readLine();
        }
        QStringList lines = QStringList();
#if defined(CO_sgi64) || defined(CO_sgin32) || defined(IRIX)
        lines = "";
#endif
        lines.append(commentString);

        for (QStringList::Iterator it = lines.begin(); it != lines.end(); ++it)
            stream << *it << "\n";
        file.close();
    }
}

/*------------------------------------------------------------------------------
 ** QString ConfigFileIO::createWinName(Window* win)
 **     creates a name for the window, looking, which projection areas it serves.
-------------------------------------------------------------------------------*/
QString ConfigFileIO::createWinName(Window *win)
{
    QString winName;
    ChannelMap *chMap = win->getChannelMap();
    ChannelMap::Iterator chIt;
    ProjectionArea *pArea;
    QStringList pNameList; // list of names of projection areas
    for (chIt = chMap->begin(); chIt != chMap->end(); ++chIt)
    {
        pArea = chIt.data().getProjectionArea();
        cout << "pArea.getName: " << pArea->getName() << endl;
        if (pNameList.find(pArea->getName()) == pNameList.end())
        {
            pNameList.append(pArea->getName());
            winName.append(pArea->getName()).append("_");
        }
    }

    if (winName != QString())
    {
        // cut last "_"-Sign
        winName.truncate(winName.length() - 1);
        return winName;
    }
    else
        return win->getName();
}

/*------------------------------------------------------------------------------
 ** QString ConfigFileIO::createChannelName(Channel* ch)
 **     creates a name for the channel, looking, which projection areas it serves.
-------------------------------------------------------------------------------*/
QString ConfigFileIO::createChannelName(Channel *ch)
{
    QString chName;
    ProjectionArea *pArea;
    QStringList pNameList; // list of names of projection areas
    pArea = ch->getProjectionArea();
    if ((pArea != 0) && (pNameList.find(pArea->getName()) == pNameList.end()))
    {
        pNameList.append(pArea->getName());
        chName.append(pArea->getName()).append("_");
    }
    if (chName != QString())
    {
        // cut last "_"-Sign
        chName.truncate(chName.length() - 1);
        return chName;
    }
    else
        return ch->getName();
}
