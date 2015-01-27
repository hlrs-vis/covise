/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** configfileio.h
 ** 2004-02-04, Matthias Feurer
 ****************************************************************************/

#ifndef CONFIGFILEIO_H
#define CONFIGFILEIO_H

#include "projectionarea.h"
#include "host.h"
#include "covergeneral.h"
#include "tracking.h"
//#include <iostream>
#include "covise.h"

#include <qstringlist.h>

class ConfigFileIO
{

public:
    ConfigFileIO();
    HostMap *getHostMap();
    ProjectionAreaMap *getProjMap();
    CoverGeneral *getGeneralSettings();
    Tracking *getTracking();

    void setHostMap(HostMap *hm);
    void setProjMap(ProjectionAreaMap *pm);
    void setGeneralSettings(CoverGeneral *gs);
    void setTracking(Tracking *t);

    bool saveConfigFile(QString name);
    //  void loadConfigFile(QString fileName,
    //		      HostMap* hm,
    //		      CoverGeneral* gc,
    //		      ProjectionAreaMap* pm);

private:
    void writeCoverConfigGeneral();
    void writeTrackerConfig();
    void writeCoverConfigHost(Host *h);
    void writeMultiPCConfig();
    void writePipeConfig(Host *h);
    void writeWindowConfig(Host *h);
    void writeChannelAndScreenConfig(Host *h);
    void writeSection(QString sectionName,
                      QStringList hostList,
                      QStringList valueList);

    void writeCommentLine(QString commentString);
    QString createWinName(Window *win);
    QString createChannelName(Channel *ch);

    HostMap *hostMap;
    ProjectionAreaMap *projMap;
    QString fileName;
    CoverGeneral *genSets;
    Tracking *tracking;
};
#endif // CONFIGFILEIO
