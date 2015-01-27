/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coFunctionEditor.h"
#include "coDefaultFunctionEditor.h"
#include "VolumePlugin.h"
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>

#include <OpenVRUI/sginterface/vruiHit.h>

using namespace vrui;
using namespace opencover;

/// Constructor
coFunctionEditor::coFunctionEditor(const char *collaborativeUIName)
    : vruiCollabInterface(VolumeCoim.get(), collaborativeUIName, vruiCollabInterface::FunctionEditor)
    , numChannels(1)
    , activeChannel(0)
{
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    dropHandle = new coPopupHandle(collaborativeUIName);
}

/// Destructor
coFunctionEditor::~coFunctionEditor()
{
    delete dropHandle;
    delete panel;
}

void coFunctionEditor::setNumChannels(int chan)
{
    numChannels = chan;
}

int coFunctionEditor::getNumChannels() const
{
    return numChannels;
}

void coFunctionEditor::setActiveChannel(int chan)
{
    if (chan > numChannels)
        chan = 0;
    if (chan < -1)
        chan = -1;
    activeChannel = chan;
}

int coFunctionEditor::getActiveChannel() const
{
    return activeChannel;
}

void coFunctionEditor::setInstantMode(bool)
{
}

void coFunctionEditor::setMin(float)
{
}

void coFunctionEditor::setMax(float)
{
}

float coFunctionEditor::getMin()
{
    return 0;
}

float coFunctionEditor::getMax()
{
    return 1;
}

void coFunctionEditor::putUndoBuffer()
{
}

void coFunctionEditor::updatePinList()
{
}

void coFunctionEditor::updateColorBar()
{
}

void coFunctionEditor::setDiscreteColors(int)
{
}

void coFunctionEditor::show()
{
    dropHandle->setVisible(true);
}

void coFunctionEditor::hide()
{
    dropHandle->setVisible(false);
}

coPin *coFunctionEditor::getCurrentPin()
{
    return NULL;
}

// Update function editor panel
void coFunctionEditor::update()
{
    dropHandle->update();
}

void coFunctionEditor::updateVolume()
{
}

void coFunctionEditor::remoteOngoing(const char *message)
{
    (void)message;

    /* switch(message[0])
   {

      default:
      {
         cerr << "coFunctionEditor: Unknown remote command" << endl;
      }
      break;
   }*/
}

coUndoValuePoti::coUndoValuePoti(const char *bt, coValuePotiActor *actor,
                                 const char *bg, coCOIM *c, const char *idName)
    : coValuePoti(bt, actor, bg, c, idName)
{
    editor = (coFunctionEditor *)actor;
}

int coUndoValuePoti::hit(vruiHit *hit)
{
    // Check for first button press:
    if (interactionA->wasStarted() || interactionB->wasStarted())
    {
        editor->putUndoBuffer();
    }
    return coValuePoti::hit(hit);
}

coUndoSlopePoti::coUndoSlopePoti(const char *bt, coValuePotiActor *actor,
                                 const char *bg, coCOIM *c, const char *idName)
    : coSlopePoti(bt, actor, bg, c, idName)
{
    editor = (coFunctionEditor *)actor;
}

int coUndoSlopePoti::hit(vruiHit *hit)
{
    // Check for first button press:
    if (interactionA->wasStarted() || interactionB->wasStarted())
    {
        editor->putUndoBuffer();
    }
    return coSlopePoti::hit(hit);
}
