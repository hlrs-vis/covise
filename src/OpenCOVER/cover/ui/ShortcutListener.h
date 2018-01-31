#ifndef COVER_UI_SHORTCUT_LISTENER_H
#define COVER_UI_SHORTCUT_LISTENER_H

#include "Export.h"
#include <string>
#include <vector>

namespace opencover {
namespace ui {

enum Modifiers {
    ModNone=0,
    ModAlt=1,
    ModCtrl=2,
    ModShift=4,
    ModMeta=8
};

enum MouseButton {
    Left = 1,
    Middle = 2,
    Right = 4,
    ScrollUp = 8,
    ScrollDown = 16,
    ScrollLeft = 32,
    ScrollRight = 64,
};

//! mix-in class for UI \ref Element "elements" reacting to keyboard shortcuts
class COVER_UI_EXPORT ShortcutListener
{
 public:
   void clearShortcuts();
   void setShortcut(const std::string &shortcut);
   void addShortcut(const std::string &shortcut);
   bool hasShortcut() const;
   bool matchShortcut(int mod, int sym) const;
   bool matchButton(int mod, int button) const;

   size_t shortcutCount() const;
   std::string shortcutText(size_t idx) const;

   virtual void shortcutTriggered();

 private:
   struct Shortcut
   {
       std::string text;
       int modifiers=ModNone;
       int symbol=0;
       int button=0;
   };
   std::vector<Shortcut> m_shortcuts;
};

}
}
#endif
