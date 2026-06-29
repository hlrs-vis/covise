#pragma once
#include "IDrawable.h"

namespace core::interface {
template <typename Info>
class IInformable : public IDrawable {
 public:
  virtual ~IInformable() = default;
  virtual void showInfo() = 0;
  virtual void hideInfo() = 0;
  virtual void updateInfo(const Info &info) = 0;

 protected:
  Info m_info;
};
}  // namespace core::interface
