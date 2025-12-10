#pragma once
namespace core::interface {
class IUpdateable {
 public:
  virtual ~IUpdateable() = default;
  IUpdateable(const IUpdateable&) = delete;
  IUpdateable operator=(const IUpdateable&) = delete;
  virtual void update() = 0;
};
}  // namespace core::interface
