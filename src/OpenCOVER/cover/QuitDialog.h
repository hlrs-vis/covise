#ifndef COVER_QUIT_DIALOG_H
#define COVER_QUIT_DIALOG_H

#include <OpenVRUI/coMenu.h>

#include "Deletable.h"

namespace vrui
{
class coRowMenu;
class coButtonMenuItem;
}

namespace opencover
{

class QuitDialog: public Deletable, public vrui::coMenuListener
{
public:
    QuitDialog();
    ~QuitDialog();

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
