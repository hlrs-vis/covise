/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READ_COMMAND_PLUGIN_H
#define READ_COMMAND_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2007 HLRS  **
 **                                                                          **
 ** Description: ReadCommand Plugin                                          **
 **                                                                          **
 **                                                                          **
 ** Author: A. Kopecki                                                       **
 **                                                                          **
 ** History                                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <QThread>
#include <QMutex>
#include <QStringList>

class QTextIStream;

class ReadCommandPlugin : public coVRPlugin, public QThread
{
public:
    ReadCommandPlugin();
    ~ReadCommandPlugin();

    void preFrame();

    virtual void run();

private:
    bool keepRunning;
    QMutex lock;
    QStringList queue;
};
#endif
