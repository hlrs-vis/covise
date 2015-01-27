/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DYNAMIC_UI_H
#define DYNAMIC_UI_H
/****************************************************************************\ 
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: Dynamic UI                                                  **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:  								     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <QMap>
#include <cover/coTabletUI.h>
#include <cover/coVRIOReader.h>

#include <wslib/WSCOVISEClient.h>

#include <osg/Group>

#include "ScriptWsCovise.h"
#include "ScriptEngineProvider.h"

using namespace opencover;

class DynamicUI : public QObject, public coVRIOReader
{

    Q_OBJECT

public:
    DynamicUI(ScriptEngineProvider *plugin, ScriptWsCovise *covise);
    virtual ~DynamicUI();

    virtual bool canLoadParts() const
    {
        return false;
    }
    virtual bool canUnload() const
    {
        return true;
    }
    virtual bool inLoading() const
    {
        return this->loading;
    }
    virtual bool abortIO()
    {
        return false;
    }
    virtual std::string getIOHandlerName() const
    {
        return "DynamicUI";
    }

    virtual osg::Node *load(const std::string &location, osg::Group *group = 0);
    virtual osg::Node *getLoaded();
    virtual bool unload(osg::Node *node);

    virtual void preFrame();

private slots:
    void tabletUICommand(const QString &target, const QString &command);

private:
    QMap<osg::Group *, coTUIUITab *> uiTabs;
    bool loading;

    ScriptWsCovise *client;
    ScriptEngineProvider *provider;
};
#endif
