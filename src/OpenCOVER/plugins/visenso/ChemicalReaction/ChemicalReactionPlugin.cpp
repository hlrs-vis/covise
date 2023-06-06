/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: ChemicalReactionPlugin                                    **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <grmsg/coGRKeyWordMsg.h>
#include <net/message.h>
#include <cover/coVRFileManager.h>

#include "cover/coTranslator.h"

#include "ChemicalReactionPlugin.h"

using namespace grmsg;

ChemicalReactionPlugin *ChemicalReactionPlugin::plugin = NULL;

ChemicalReactionPlugin::ChemicalReactionPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, GenericGuiObject("ChemicalReaction")
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\nChemicalReactionPlugin::ChemicalReactionPlugin\n");
}

ChemicalReactionPlugin::~ChemicalReactionPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nChemicalReactionPlugin::~ChemicalReactionPlugin\n");
}

bool ChemicalReactionPlugin::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(0))
        fprintf(stderr, "\nChemicalReactionPlugin::init\n");

    // set plugin
    ChemicalReactionPlugin::plugin = this;

    p_nextPresStepAllowed = addNextPresStepAllowed(true);
    p_startMolecule[0] = addGuiParamString("StartMolecule1", "");
    p_startMolecule[1] = addGuiParamString("StartMolecule2", "");
    p_startMolecule[2] = addGuiParamString("StartMolecule3", "");
    p_endMolecule = addGuiParamString("EndMolecule", "");

    moleculeHandler = new MoleculeHandler();
    reactionArea = new ReactionArea();
    startButton = new StartButton();

    indicatorTransform = new osg::MatrixTransform();
    indicatorTransform->setMatrix(osg::Matrix::translate(-7.2f, -0.1f, -7.2f));
    cover->getObjectsRoot()->addChild(indicatorTransform.get());
    geometryOk = coVRFileManager::instance()->loadIcon("atombaukasten/button_ok");
    geometryNotOk = coVRFileManager::instance()->loadIcon("atombaukasten/button_notok");

    hud = coHud::instance();
    hudTime = 0.0f;

    return true;
}

void ChemicalReactionPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    GenericGuiObject::guiToRenderMsg(msg);

    if (msg.isValid() && msg.getType() == coGRMsg::KEYWORD)
    {
        auto &keywordmsg = msg.as<coGRKeyWordMsg>();
        const char *keyword = keywordmsg.getKeyWord();
        if (strcmp(keyword, "showNotReady") == 0)
        {
            hud->setText1(coTranslator::coTranslate("Die Antwort ist nicht korrekt! \n Versuchen Sie es noch einmal.").c_str());
            hud->show();
            hud->redraw();
            if (hudTime == 0.0f)
                hudTime = 0.001f;
        }
    }
}

void ChemicalReactionPlugin::preFrame()
{
    moleculeHandler->preFrame();

    if (hudTime > 2.5f)
    {
        hudTime = 0.0f;
        hud->hide();
    }
    else if (hudTime > 0.0f)
    {
        hudTime += cover->frameDuration();
    }

    if (startButton->wasClicked())
    {
        if (moleculeHandler->getState() == STATE_DEFAULT)
        {
            moleculeHandler->performReaction();
            p_nextPresStepAllowed->setValue(moleculeHandler->isCorrect(p_endMolecule->getValue()));
            if (moleculeHandler->getState() != STATE_DEFAULT)
                startButton->setButtonState(BUTTON_STATE_RESET);
        }
        else
        {
            moleculeHandler->resetReaction();
            p_nextPresStepAllowed->setValue(false);
            startButton->setButtonState(BUTTON_STATE_START);
        }
        updateOkIndicator();
    }
}

void ChemicalReactionPlugin::guiParamChanged(GuiParam *guiParam)
{
    if ((guiParam == p_startMolecule[0]) || (guiParam == p_startMolecule[1]) || (guiParam == p_startMolecule[2]) || (guiParam == p_endMolecule))
    {
        validTask = (p_startMolecule[0]->getValue().compare("") != 0)
                    || (p_startMolecule[1]->getValue().compare("") != 0)
                    || (p_startMolecule[2]->getValue().compare("") != 0);

        moleculeHandler->clear();
        for (int i = 0; i < 3; ++i)
        {
            moleculeHandler->createStartMolecules(i, p_startMolecule[i]->getValue());
        }

        startButton->setButtonState(BUTTON_STATE_START);
        p_nextPresStepAllowed->setValue(!validTask);
        startButton->setVisible(validTask);
        reactionArea->setVisible(validTask);
        updateOkIndicator();
    }
}

void ChemicalReactionPlugin::updateOkIndicator()
{
    bool okVisible = validTask && (moleculeHandler->getState() != STATE_DEFAULT) && moleculeHandler->isCorrect(p_endMolecule->getValue());
    bool notOkVisible = validTask && (moleculeHandler->getState() != STATE_DEFAULT) && !moleculeHandler->isCorrect(p_endMolecule->getValue());

    if (indicatorTransform->containsNode(geometryOk.get()))
    {
        if (!okVisible)
            indicatorTransform->removeChild(geometryOk.get());
    }
    else
    {
        if (okVisible)
            indicatorTransform->addChild(geometryOk.get());
    }

    if (indicatorTransform->containsNode(geometryNotOk.get()))
    {
        if (!notOkVisible)
            indicatorTransform->removeChild(geometryNotOk.get());
    }
    else
    {
        if (notOkVisible)
            indicatorTransform->addChild(geometryNotOk.get());
    }
}

COVERPLUGIN(ChemicalReactionPlugin)
