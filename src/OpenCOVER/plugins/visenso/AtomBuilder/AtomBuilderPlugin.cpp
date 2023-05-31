/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: AtomBuilderPlugin                                         **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: C. Spenrath, D. Rainer                                         **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include <covise/covise_msg.h>
#include <grmsg/coGRKeyWordMsg.h>

#include "cover/coTranslator.h"

#include "AtomBuilderPlugin.h"
#include "ElementDatabase.h"
#include "AtomBuilder.h"

using namespace grmsg;

AtomBuilderPlugin *AtomBuilderPlugin::plugin = NULL;

AtomBuilderPlugin::AtomBuilderPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, GenericGuiObject("AtomBuilder")
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\nAtomBuilderPlugin::AtomBuilderPlugin\n");
}

AtomBuilderPlugin::~AtomBuilderPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nAtomBuilderPlugin::~AtomBuilderPlugin\n");

    delete atomBuilder;
}

bool AtomBuilderPlugin::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(0))
        fprintf(stderr, "\nAtomBuilderPlugin::init\n");

    // set plugin
    AtomBuilderPlugin::plugin = this;

    p_element = addGuiParamString(coTranslator::coTranslate("Element"), "");
    p_nextPresStepAllowed = addNextPresStepAllowed(true);
    p_puzzleMode = addGuiParamBool("ShowAtomBuilder", false);

    // create atom builder
    atomBuilder = new AtomBuilder();

    return true;
}

void AtomBuilderPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    GenericGuiObject::guiToRenderMsg(msg);

    if (msg.isValid() && msg.getType() == coGRMsg::KEYWORD)
    {
        auto &keywordmsg = msg.as<coGRKeyWordMsg>();
        const char *keyword = keywordmsg.getKeyWord();
        fprintf(stderr, "AtomBuilderPlugin::guiToRenderMsg msg=%s\n", keyword);
        if (strcmp(keyword, "showNotReady") == 0)
        {
            atomBuilder->showErrorPanel();
        }
    }
}

void AtomBuilderPlugin::preFrame()
{
    atomBuilder->update();
    p_nextPresStepAllowed->setValue(atomBuilder->getStatus());
}

void AtomBuilderPlugin::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_element)
    {
        Element element = ElementDatabase::Instance()->findBySymbol(p_element->getValue());
        cerr << element.name << endl;

        // set new configuration
        atomBuilder->setElement(element);
    }
    else if (guiParam == p_puzzleMode)
    {
        atomBuilder->show(p_puzzleMode->getValue());
    }
}

COVERPLUGIN(AtomBuilderPlugin)
