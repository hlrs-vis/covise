#ifndef UI_OWNER_H
#define UI_OWNER_H

#include <string>
#include <map>

#include "Export.h"

namespace opencover {
namespace ui {

class Manager;

//! Manage the life time of objects: in its destructor it destroys all objects that it owns
class COVER_UI_EXPORT Owner {
   friend class Manager;
   friend class View;
 public:
   //! construct, owned by owner (deleted with owner)
   Owner(const std::string &name, Owner *owner);
   //! construct without owner: must be deleted
   Owner(const std::string &name, Manager *manager);
   Owner() = delete;
   virtual ~Owner();
   //! return owner
   Owner *owner() const;
   //! return name of object
   const std::string &name() const;
   //! return path to object through all owners
   std::string path() const;
   //! return manager responsible for this item
   Manager *manager() const;

   static bool checkName(const std::string &name);
   static std::string makeName(const std::string &name);

 protected:
   void clearItems();

 private:
   bool addItem(Owner *item);
   bool removeItem(Owner *item);

   const std::string m_name;
   Owner *m_owner = nullptr;
   Manager *m_manager = nullptr;
   std::map<std::string, Owner *> m_items;
};

}
}
#endif
