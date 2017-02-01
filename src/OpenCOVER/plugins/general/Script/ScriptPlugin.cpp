/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "ScriptPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include <QFile>
#include <QTextStream>

ScriptPlugin *ScriptPlugin::plugin = NULL;
static FileHandler handlers[] = {
    { NULL,
      ScriptPlugin::loadQS,
      ScriptPlugin::loadQS,
      ScriptPlugin::unloadQS,
      "qs" },
    { NULL,
      ScriptPlugin::loadQS,
      ScriptPlugin::loadQS,
      ScriptPlugin::unloadQS,
      "js" }
};

ScriptPlugin::ScriptPlugin()
{
    plugin = this;
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    fprintf(stderr, "ScriptPlugin::ScriptPlugin\n");

    std::string startupScript = coCoviseConfig::getEntry("COVER.Plugin.Script.StartupScript");
    if (!startupScript.empty())
    {
        loadScript(startupScript);
    }
    CommandLine = new coTUIEditField("cover.setVisible(\"Sphere01-FACES\",true);", coVRTui::instance()->getCOVERTab()->getID());
    CommandLine->setPos(1, 13);
    CommandLine->setEventListener(this);
    CommandLineLabel = new coTUILabel("CommandLine", coVRTui::instance()->getCOVERTab()->getID());
    CommandLineLabel->setPos(0, 13);
    myCOVER = new ScriptInterface(this);
    QScriptValue objectValue = engine.newQObject(myCOVER);
    engine.globalObject().setProperty("cover", objectValue);
}

// this is called if the plugin is removed at runtime
ScriptPlugin::~ScriptPlugin()
{
    fprintf(stderr, "ScriptPlugin::~ScriptPlugin\n");
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
    delete CommandLine;
    delete CommandLineLabel;
    delete myCOVER;
}

int ScriptPlugin::unloadQS(const char *filename, const char *)
{
    std::string filen;
    if (filename)
    {
        filen = filename;
    }
    return plugin->unloadScript(filen);
}

int ScriptPlugin::unloadScript(std::string /*filename*/)
{
    return 1;
}

int ScriptPlugin::loadQS(const char *filename, osg::Group * /*loadParent*/, const char *)
{
    std::string filen;
    if (filename)
    {
        filen = filename;
    }
    return plugin->loadScript(filen);
}

int ScriptPlugin::loadScript(std::string scriptName)
{
    QString fileName = scriptName.c_str();
    QFile scriptFile(fileName);
    if (!scriptFile.open(QIODevice::ReadOnly))
    {
        return 0;
    }
    QTextStream stream(&scriptFile);
    QString contents = stream.readAll();
    scriptFile.close();
    engine.evaluate(contents, fileName);

    return 1;
}

void ScriptPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == CommandLine)
    {
        engine.evaluate(CommandLine->getText());
    }
}

void
ScriptPlugin::preFrame()
{
}

COVERPLUGIN(ScriptPlugin)
