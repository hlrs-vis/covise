#pragma once
#include "IColorable.h"
#include "IDrawable.h"

namespace core::interface {
class ISolarPanel : public IDrawable, public IColorable {
 public:
  virtual ~ISolarPanel() = default;
};
}  // namespace core::interface
