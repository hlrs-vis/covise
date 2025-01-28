#pragma once

#include "Export.h"
#include <string>
#include <vector>
#include <vsg/ui/KeyEvent.h>

namespace vive {
namespace ui {

enum Modifiers { // same as vsg::MODKEY_
    ModNone=0,
    ModShift = 1,
    ModCtrl=4,
    ModAlt = 8,
    ModMeta=128
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
class VIVE_UI_EXPORT ShortcutListener
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
       std::string key;
       int modifiers = ModNone;
       int symbol=0;
       int button=0;
   };
   std::vector<Shortcut> m_shortcuts;
};

}
}
