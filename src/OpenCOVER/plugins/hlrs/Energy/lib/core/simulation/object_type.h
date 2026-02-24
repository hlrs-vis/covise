#pragma once

namespace core::simulation {

enum class ObjectType {
  Bus,
  Generator,
  Transformator,
  Cable,
  Building,
  Consumer,
  Producer,
  // Add more types as needed
  Unknown
};

}