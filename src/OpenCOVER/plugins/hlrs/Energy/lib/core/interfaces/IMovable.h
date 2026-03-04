#pragma once
#include <array>

namespace core::interface {
struct Pos {
    double x, y, z;
    Pos() : x(0.0f), y(0.0f), z(0.0f) {}
    Pos(double x, double y, double z) : x(x), y(y), z(z) {}
    Pos(const float xyz[3])
        : x(xyz[0])
        , y(xyz[1])
        , z(xyz[2])
    {
    }
    Pos(const std::array<float, 4> &xyz)
        : x(xyz[0])
        , y(xyz[1])
        , z(xyz[2])
    {
    }
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
