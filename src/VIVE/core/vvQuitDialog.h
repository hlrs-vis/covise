#ifndef COVER_QUIT_DIALOG_H
#define COVER_QUIT_DIALOG_H

#include <OpenVRUI/coMenu.h>

#include "vvDeletable.h"

namespace vrui
{
class coRowMenu;
class coButtonMenuItem;
}

namespace vive
{

class vvQuitDialog: public vvDeletable, public vrui::coMenuListener
{
public:
    vvQuitDialog();
    ~vvQuitDialog();

    void show();

private:
    void menuEvent(vrui::coMenuItem *item) override;

    void init();
    void hide();

    vrui::coRowMenu *quitMenu_ = nullptr;
    vrui::coButtonMenuItem *yesButton_ = nullptr;
    vrui::coButtonMenuItem *cancelButton_ = nullptr;
};

}
#endif
