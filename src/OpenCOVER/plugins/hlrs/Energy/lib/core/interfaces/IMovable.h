#pragma once

namespace core::interface {
struct Pos {
    float x, y, z;
    Pos() : x(0.0f), y(0.0f), z(0.0f) {}
    Pos(float x, float y, float z) : x(x), y(y), z(z) {}
};

class IMoveable {
 public:
  IMoveable() = default;
  virtual ~IMoveable() = default; // only for polymorphe destruction and not for ressource management
  IMoveable(const IMoveable&) = delete;
  IMoveable& operator=(const IMoveable&) = delete;
  virtual void move(const Pos &pos) = 0;
};
}  // namespace core::interface
