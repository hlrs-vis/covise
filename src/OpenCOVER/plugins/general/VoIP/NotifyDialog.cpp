/* This file is part of COVISE.
   
   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "NotifyDialog.h"

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <string>
#include <algorithm>

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/OpenCOVER.h>
//#include "coTranslator.h"

using covise::coCoviseConfig;
using namespace vrui;

namespace opencover
{

// ----------------------------------------------------------------------------
NotifyDialog::NotifyDialog()
{
    init();
}

// ----------------------------------------------------------------------------
NotifyDialog::~NotifyDialog()
{
    delete lmi2ndLine;
    delete lmiQuestion;
    delete btnLeft;
    delete btnRight;
    delete menuNotify;
}

// ----------------------------------------------------------------------------
void NotifyDialog::init()
{
    strQuestion = "YES OR NO?";
    str2ndLine = " ";
    strLeftOption = "yes";
    strRightOption = "no";

    menuNotify = new vrui::coRowMenu("NOTIFY");
    menuNotify->setVisible(false);
    menuNotify->setAttachment(coUIElement::RIGHT);

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

    menuNotify->setTransformMatrix(&matrix);
    menuNotify->setScale(0.5); //cover->getSceneSize() / 2500);

    lmiQuestion = new coLabelMenuItem(strQuestion.c_str());
    menuNotify->add(lmiQuestion);
    
    lmi2ndLine = new coLabelMenuItem(str2ndLine.c_str());
    menuNotify->add(lmi2ndLine);
        
    btnLeft = new coButtonMenuItem(strLeftOption.c_str());
    btnLeft->setMenuListener(this);
    menuNotify->add(btnLeft);
    
    btnRight = new coButtonMenuItem(strRightOption.c_str());
    btnRight->setMenuListener(this);
    menuNotify->add(btnRight);
}

// ----------------------------------------------------------------------------
void NotifyDialog::setText(std::string _strQuestion, std::string _strLeft, std::string _strRight)
{
    strQuestion = _strQuestion;
    strLeftOption = _strLeft;
    strRightOption = _strRight;

    lmiQuestion->setName(strQuestion.c_str());
    btnLeft->setName(strLeftOption.c_str());
    btnRight->setName(strRightOption.c_str());
}

// ----------------------------------------------------------------------------
std::string NotifyDialog::getSelection()
{
    return strSelection;
}

// ----------------------------------------------------------------------------
void NotifyDialog::show()
{
    menuNotify->setVisible(true);
    strSelection = "";
}

// ----------------------------------------------------------------------------
void NotifyDialog::hide()
{
    menuNotify->setVisible(false);
}

// ----------------------------------------------------------------------------
void NotifyDialog::menuEvent(coMenuItem *item)
{
    if (item == btnLeft)
    {
        hide();
        strSelection = strLeftOption;
    }
    else if (item == btnRight)
    {
        hide();
        strSelection = strRightOption;
    }
}

// ----------------------------------------------------------------------------
    
}
