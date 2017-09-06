/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_PLUGIN_H
#define COVISE_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2013 HLRS  **
 **                                                                          **
 ** Description: COVISE Plugin (COVIS module interface)                      **
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumüller                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

using namespace opencover;

class CovisePlugin : public coVRPlugin, public ui::Owner
{
public:
    CovisePlugin();
    ~CovisePlugin();

    bool init();
    void notify(NotificationLevel level, const char *text);
    void param(const char *paramName, bool inMapLoading);
    bool update();
    void preFrame();
    void removeNode(osg::Node *group, bool isGroup, osg::Node *node);
    void requestQuit(bool killSession);
    bool sendVisMessage(const covise::Message *msg);
    bool becomeCollaborativeMaster();
    bool executeAll();
    covise::Message *waitForVisMessage(int type);
    void expandBoundingSphere(osg::BoundingSphere &bs);
};
#endif
