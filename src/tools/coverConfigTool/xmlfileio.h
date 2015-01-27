/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** xmlfileio.h
 ** 2004-02-05, Matthias Feurer
 ****************************************************************************/

#ifndef XMLFILEIO_H
#define XMLFILEIO_H

#include "projectionarea.h"
#include "host.h"
#include "covergeneral.h"
#include "tracking.h"
#include <qdom.h>

#include <qstringlist.h>

class XMLFileIO
{

public:
    XMLFileIO();
    HostMap *getHostMap();
    ProjectionAreaMap *getProjMap();
    CoverGeneral *getGeneralSettings();
    Tracking *getTracking();

    void setHostMap(HostMap *hm);
    void setProjMap(ProjectionAreaMap *pm);
    void setGeneralSettings(CoverGeneral *gs);
    void setTracking(Tracking *t);

    bool saveXMLFile(QString name);
    bool loadXMLFile(QString name,
                     QString *message);

private:
    void traverseNode(const QDomNode &node);

    HostMap *hostMap;
    ProjectionAreaMap *projMap;
    CoverGeneral *genSets;
    Tracking *tracking;
    QString fileName;
    QDomDocument domDoc;
};
#endif // XMLFILEIO_H
