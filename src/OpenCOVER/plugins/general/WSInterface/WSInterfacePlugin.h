/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WS_INTERFACE_PLUGIN_H
#define WS_INTERFACE_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: WSInterface Plugin                                          **
 **                                                                          **
 **                                                                          **
 ** Author: M. Baalcke                                                      **
 **                                                                          **
 ** History                                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <QMutex>
#include <QStringList>

class QTextIStream;

class WSInterfacePlugin : public coVRPlugin
{
public:
    WSInterfacePlugin();
    ~WSInterfacePlugin();
    static WSInterfacePlugin *instance();

    void preFrame();

    void openFile(const QString &filename);
    void addFile(const QString &filename);
    void quit();
    void connectToVnc(const QString &host, unsigned int port, const QString &password);
    void disconnectFromVnc();
    void setVisibleVnc(bool on);
    void sendCustomMessage(const QString &parameter);

    void show(const QString &name);
    void hide(const QString &name);

    void viewAll();
    void resetView();
    void walk();
    void fly();
    void drive();
    void scale();
    void xform();
    void wireframe(bool on);

    void snapshot(const QString &path);

    enum MessageType
    {
        CustomMessage = 16532
    };

    //virtual void run();

private:
    //bool keepRunning;
    QMutex lock;
    static WSInterfacePlugin *singleton;

    QStringList queue;

    void setVisible(const QString &name, bool on);
};
#endif
