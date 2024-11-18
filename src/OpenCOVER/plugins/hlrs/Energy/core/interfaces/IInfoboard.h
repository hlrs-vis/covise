#ifndef _CORE_INTERFACES_IINFOBOARD_H
#define _CORE_INTERFACES_IINFOBOARD_H

#include "IInformable.h"
#include "IMovable.h"
#include "ITimedependable.h"

namespace core {
namespace interface {
template <typename Info>
class IInfoboard : public IInformable<Info>,
                   public ITimedependable,
                   public IMoveable {
 public:
  virtual void initInfoboard() = 0;
  bool enabled() { return m_enabled; }

 protected:
  bool m_enabled = false;
};
}  // namespace interface
}  // namespace core

#endif
