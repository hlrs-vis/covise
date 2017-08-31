#include "QuitDialog.h"

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <string>
#include <algorithm>

#include <config/CoviseConfig.h>
#include "coVRPluginSupport.h"
#include "OpenCOVER.h"
#include "coTranslator.h"

using covise::coCoviseConfig;
using namespace vrui;

namespace opencover
{

QuitDialog::QuitDialog()
{
    init();
}

QuitDialog::~QuitDialog()
{
    delete quitMenu_;
    delete yesButton_;
    delete cancelButton_;
}

void QuitDialog::init()
{
    std::string qtext, yesText, noText;

    qtext = "Really quit OpenCOVER?";
    yesText = "Quit";
    noText = "Continue";

    qtext = coTranslator::coTranslate(qtext);
    yesText = coTranslator::coTranslate(yesText);
    noText = coTranslator::coTranslate(noText);

    quitMenu_ = new coRowMenu(qtext.c_str());
    quitMenu_->setVisible(false);
    quitMenu_->setAttachment(coUIElement::RIGHT);
    OSGVruiMatrix transMatrix, scaleMatrix, rotateMatrix, matrix;

    float px = coCoviseConfig::getFloat("x", "COVER.QuitMenu.Position", 0.0);
    float py = coCoviseConfig::getFloat("y", "COVER.QuitMenu.Position", 0.0);
    float pz = coCoviseConfig::getFloat("z", "COVER.QuitMenu.Position", 0.0);
    float s = coCoviseConfig::getFloat("value", "COVER.QuitMenu.Size", 1.0f);

    matrix.makeIdentity();
    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    quitMenu_->setTransformMatrix(&matrix);
    quitMenu_->setScale(cover->getSceneSize() / 2500);
    yesButton_ = new coButtonMenuItem(yesText.c_str());
    yesButton_->setMenuListener(this);
    cancelButton_ = new coButtonMenuItem(noText.c_str());
    cancelButton_->setMenuListener(this);
    quitMenu_->add(yesButton_);
    quitMenu_->add(cancelButton_);
}

void QuitDialog::show()
{
    quitMenu_->setVisible(true);
}

void QuitDialog::hide()
{
    quitMenu_->setVisible(false);

    deleteLater();
}


void QuitDialog::menuEvent(coMenuItem *item)
{
    if (item == yesButton_)
    {
        hide();
        OpenCOVER::instance()->quitCallback(NULL, NULL);
    }
    else if (item == cancelButton_)
    {
        hide();
    }
}

}
