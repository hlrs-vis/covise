#ifndef UI_LABEL_H
#define UI_LABEL_H

#include "Element.h"

namespace opencover {
namespace ui {

//! a graphical Element showing a text label

/** \note QLabel
    \note vrui::coLabelMenuItem
    \note coTUILabel
    */
class COVER_UI_EXPORT Label: public Element {

 public:
   Label(Group *parent, const std::string &name);
   Label(const std::string &name, Owner *owner);
   ~Label();
};

}
}
#endif
