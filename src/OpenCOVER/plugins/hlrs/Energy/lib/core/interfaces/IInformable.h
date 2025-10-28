#pragma once
#include "IDrawable.h"

namespace core::interface {
template <typename InfoType>
class IInformable : public IDrawable {
 public:
  virtual ~IInformable() = default;
  virtual void showInfo() = 0;
  virtual void hideInfo() = 0;
  virtual void updateInfo(const InfoType &info) = 0;
};
}  // namespace core::interface
