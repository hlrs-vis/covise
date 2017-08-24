#ifndef COVER_UI_SHORTCUT_LISTENER_H
#define COVER_UI_SHORTCUT_LISTENER_H

#include "Export.h"
#include <string>

namespace opencover {
namespace ui {

enum Modifiers {
    ModNone=0,
    ModAlt=1,
    ModCtrl=2,
    ModShift=4,
    ModMeta=8
};

class ShortcutListener
{
 public:
   void setShortcut(const std::string &shortcut);
   bool hasShortcut() const;
   int modifiers() const;
   int symbol() const;

   virtual void shortcutTriggered();

 private:
   int m_modifiers=ModNone;;
   int m_symbol=0;
};

}
}
#endif
