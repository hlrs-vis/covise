#pragma once
#include "IInformable.h"
#include "IMovable.h"
#include "ITimedependable.h"

namespace core::interface {
template <typename Info>
class IInfoboard : public IInformable<Info>,
                   public ITimedependable,
                   public IMoveable {
 public:
  virtual ~IInfoboard() = default;
  virtual void initInfoboard() = 0;
  virtual bool enabled() = 0;
};
}  // namespace core::interface
