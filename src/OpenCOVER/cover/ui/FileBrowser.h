#ifndef UI_FILEBROWSER_H
#define UI_FILEBROWSER_H

#include "Element.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical Element allowing for keyboard input

/** \note QLineEdit
    \note coTUIEditField */
class COVER_UI_EXPORT FileBrowser: public Element {

 public:
   enum UpdateMask: UpdateMaskType
   {
       UpdateValue = 0x100,
       UpdateFilter = 0x200,
   };

   FileBrowser(Group *parent, const std::string &name, bool save=false);
   FileBrowser(const std::string &name, Owner *owner, bool save=false);
   virtual ~FileBrowser();

   void setValue(const std::string &text);
   std::string value() const;

   void setFilter(const std::string &filter);
   std::string filter() const;

   void setCallback(const std::function<void(const std::string &text)> &f);
   std::function<void(const std::string &)> callback() const;

   void triggerImplementation() const override;

    void update(UpdateMaskType mask) const override;

    void save(covise::TokenBuffer &buf) const override;
    void load(covise::TokenBuffer &buf) override;

    bool forSaving() const;

 protected:
    std::function<void(const std::string &text)> m_callback;
    std::string m_value;
    std::string m_filter;
    bool m_save = false;
};

}
}
#endif
