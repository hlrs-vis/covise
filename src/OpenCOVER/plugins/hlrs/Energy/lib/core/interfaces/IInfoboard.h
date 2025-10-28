#pragma once
#include "IInformable.h"
#include "IMovable.h"

namespace core::interface {
template <typename InfoType, typename DrawableType>
class IInfoboard : public IInformable<InfoType>,
                   public IMoveable {
 public:
  virtual ~IInfoboard() = default;
  virtual void initInfoboard() = 0;
  virtual bool enabled() = 0;
  virtual DrawableType& getDrawable() = 0;
};
}  // namespace core::interface
