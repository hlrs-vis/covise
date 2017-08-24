#ifndef UI_ELEMENT_H
#define UI_ELEMENT_H

#include <string>
#include <set>
#include "Owner.h"
#include "ShortcutListener.h"

namespace opencover {
namespace ui {

class Container;
class Group;

class COVER_UI_EXPORT Element: public Owner, public ShortcutListener {
    friend class Manager;
    friend class Group;
    friend class Container;
 public:
    Element(const std::string &name, Owner *owner);
    Element(Group *group, const std::string &name);
    virtual ~Element();

    Group *parent() const;

    std::set<Container *> containers();
    virtual void update() const;

    const std::string &text() const;
    void setText(const std::string &text);

    bool visible() const;
    void setVisible(bool flag);
    bool enabled() const;
    void setEnabled(bool flag);

    void trigger() const;
 protected:
    virtual void triggerImplementation() const;
    Group *m_parent = nullptr;
    std::set<Container *> m_containers;
    std::string m_label;
    bool m_visible = true;
    bool m_enabled = true;
};

}
}
#endif
