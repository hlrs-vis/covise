#pragma once
#include "IColorable.h"
#include "IDrawable.h"

namespace core::interface {
class ISolarPanel : public IDrawable, public IColorable {
 public:
  ISolarPanel() = default;
  virtual ~ISolarPanel() = default;
  ISolarPanel(const ISolarPanel&) = delete;
  ISolarPanel& operator=(const ISolarPanel&) = delete;
};
}  // namespace core::interface
