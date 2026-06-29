#pragma once
#include "IColorable.h"
#include "IDrawables.h"

namespace prototype::core::interface {
class ISolarPanel : public IDrawables, public IColorable {
 public:
  virtual ~ISolarPanel() = default;
};
}  // namespace core::interface
