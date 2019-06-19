/* This file is part of COVISE.
   
   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _NOTIFY_DIALOG_H
#define _NOTIFY_DIALOG_H

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>

//#include "Deletable.h"

namespace opencover
{

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
class NotifyDialog: public vrui::coMenuListener
{
public:
    NotifyDialog();
    ~NotifyDialog();

    void show();
    void setText(std::string _strQuestion, std::string _strLeft, std::string _strRight);

    std::string getSelection();
    
protected:
    void menuEvent(vrui::coMenuItem *item) override;

    void init();
    void hide();
    
    vrui::coRowMenu *menuNotify = nullptr;
    vrui::coButtonMenuItem *btnLeft = nullptr;
    vrui::coButtonMenuItem *btnRight = nullptr;

    vrui::coLabelMenuItem* lmiQuestion = nullptr;
    vrui::coLabelMenuItem* lmi2ndLine = nullptr;
    
    std::string strQuestion;
    std::string str2ndLine;;
    std::string strLeftOption;
    std::string strRightOption;

    std::string strSelection;
};

}

// ----------------------------------------------------------------------------

#endif
