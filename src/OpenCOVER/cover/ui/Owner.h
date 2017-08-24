#ifndef UI_OWNER_H
#define UI_OWNER_H

#include <string>
#include <map>

#include "Export.h"

namespace opencover {
namespace ui {

class Manager;

class COVER_UI_EXPORT Owner {
   friend class Manager;
 public:
   Owner(const std::string &name, Owner *owner);
   Owner(const std::string &name, Manager *manager);
   virtual ~Owner();
   Owner *owner() const;
   const std::string &name() const;
   std::string path() const;
   Manager *manager() const;

 private:
   void clearItems();
   bool addItem(Owner *item);
   bool removeItem(Owner *item);

   const std::string m_name;
   Owner *m_owner;
   Manager *m_manager;
   std::map<std::string, Owner *> m_items;
};

}
}
#endif
