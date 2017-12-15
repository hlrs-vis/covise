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

    bool init() override;
    void notify(NotificationLevel level, const char *text) override;
    void param(const char *paramName, bool inMapLoading) override;
    bool update() override;
    void preFrame() override;
    void removeNode(osg::Node *group, bool isGroup, osg::Node *node) override;
    void requestQuit(bool killSession) override;
    bool sendVisMessage(const covise::Message *msg) override;
    bool becomeCollaborativeMaster() override;
    bool executeAll() override;
    covise::Message *waitForVisMessage(int type) override;
    void expandBoundingSphere(osg::BoundingSphere &bs) override;
    bool requestInteraction(coInteractor *inter, osg::Node *triggerNode, bool isMouse) override;
private:

    bool m_updateNeeded = false;
};
#endif
