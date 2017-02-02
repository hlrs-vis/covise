/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SCRIPT_PLUGIN_H
#define _SCRIPT_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Script Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <QMainWindow>
#include <QString>
#include <QScriptEngine>

#include "ScriptEngineProvider.h"
#include "ScriptInterface.h"
#include <cover/coTabletUI.h>
#include "DynamicUI.h"

using namespace covise;
using namespace opencover;

class ScriptTabletUI;
class ScriptWsCovise;
class ScriptDebugger;

class ScriptPlugin : public coVRPlugin, public coTUIListener, public ScriptEngineProvider
{

public:
    ScriptPlugin();
    ~ScriptPlugin();

    static int loadQS(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadQS(const char *filename, const char *ck = "");

    int loadScript(const QString &scriptName);
    int unloadScript(const QString &scriptName);

    void evaluate(const QString &command);

    virtual void tabletEvent(coTUIElement *tUIItem);

    // this will be called in PreFrame
    void preFrame();

    bool init();

    QScriptEngine &engine();

    ScriptWsCovise *covise() const
    {
        return this->covise_;
    }

    void message(int type, int len, const void *buf);

private:
    static ScriptPlugin *plugin;
    QScriptEngine *engine_;

    coTUIEditField *commandLine;
    coTUILabel *commandLineLabel;
    ScriptInterface *myCOVER;

    ScriptTabletUI *tui;
    ScriptWsCovise *covise_;
#if QT_VERSION >= 0x040500
    ScriptDebugger *debugger;
#endif

    DynamicUI *dynamicUI;

    QApplication *app;
};
#endif
