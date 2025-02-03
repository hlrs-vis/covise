#include "vvQuitDialog.h"

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include <string>
#include <algorithm>

#include <config/CoviseConfig.h>
#include "vvPluginSupport.h"
#include "vvVIVE.h"
#include "vvTranslator.h"

using covise::coCoviseConfig;
using namespace vrui;

namespace vive
{

vvQuitDialog::vvQuitDialog()
{
    init();
}

vvQuitDialog::~vvQuitDialog()
{
    delete quitMenu_;
    delete yesButton_;
    delete cancelButton_;
}

void vvQuitDialog::init()
{
    std::string qtext, yesText, noText;

    qtext = "Really quit VIVE?";
    yesText = "Quit";
    noText = "Continue";

    qtext = vvTranslator::coTranslate(qtext);
    yesText = vvTranslator::coTranslate(yesText);
    noText = vvTranslator::coTranslate(noText);

    quitMenu_ = new coRowMenu(qtext.c_str());
    quitMenu_->setVisible(false);
    quitMenu_->setAttachment(coUIElement::RIGHT);
    VSGVruiMatrix transMatrix, scaleMatrix, rotateMatrix, matrix;

    double px = coCoviseConfig::getFloat("x", "COVER.QuitMenu.Position", 0.0);
    double py = coCoviseConfig::getFloat("y", "COVER.QuitMenu.Position", 0.0);
    double pz = coCoviseConfig::getFloat("z", "COVER.QuitMenu.Position", 0.0);
    double s = coCoviseConfig::getFloat("value", "COVER.QuitMenu.Size", 1.0f);

    matrix.makeIdentity();
    transMatrix.setMatrix(vsg::translate(px, py, pz));
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.setMatrix(vsg::scale(s, s, s));

    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    quitMenu_->setTransformMatrix(&matrix);
    quitMenu_->setScale(vv->getSceneSize() / 2500);
    yesButton_ = new coButtonMenuItem(yesText.c_str());
    yesButton_->setMenuListener(this);
    cancelButton_ = new coButtonMenuItem(noText.c_str());
    cancelButton_->setMenuListener(this);
    quitMenu_->add(yesButton_);
    quitMenu_->add(cancelButton_);
}

void vvQuitDialog::show()
{
    quitMenu_->setVisible(true);
}

void vvQuitDialog::hide()
{
    quitMenu_->setVisible(false);

    deleteLater();
}


void vvQuitDialog::menuEvent(coMenuItem *item)
{
    if (item == yesButton_)
    {
        hide();
        vvVIVE::instance()->requestQuit();
    }
    else if (item == cancelButton_)
    {
        hide();
    }
}

}
