#pragma once
namespace core::interface {
class IUpdateable {
 public:
  virtual ~IUpdateable() = default;
  virtual void update() = 0;
};
}  // namespace core::interface
