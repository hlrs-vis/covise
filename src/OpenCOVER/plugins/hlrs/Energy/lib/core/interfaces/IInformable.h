#pragma once
#include "IDrawable.h"

namespace core::interface {
template <typename InfoType>
class IInformable : public IDrawable {
 public:
  virtual ~IInformable() = default;
  IInformable(const IInformable&) = delete;
  IInformable& operator=(const IInformable&) = delete;
  virtual void showInfo() = 0;
  virtual void hideInfo() = 0;
  virtual void updateInfo(const InfoType &info) = 0;
};
}  // namespace core::interface
