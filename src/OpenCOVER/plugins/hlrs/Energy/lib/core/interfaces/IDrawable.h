#pragma once

namespace core::interface {
class IDrawable {
 public:
  IDrawable() = default;
  virtual void initDrawable() = 0;
  virtual ~IDrawable() = default;
  IDrawable(const IDrawable&) = delete;
  IDrawable(IDrawable&&) = delete;
  IDrawable& operator=(const IDrawable&) = delete;
  IDrawable& operator=(IDrawable&&) = delete;
};
}  // namespace core::interface
