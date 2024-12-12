#ifndef _CORE_INTERFACES_ISEPARABLE_H
#define _CORE_INTERFACES_ISEPARABLE_H

namespace core {
namespace interface {
class ISeparable {
 public:
  virtual ~ISeparable() = default;
  virtual void seperate() = 0;
};
}  // namespace interface
}  // namespace core

#endif
