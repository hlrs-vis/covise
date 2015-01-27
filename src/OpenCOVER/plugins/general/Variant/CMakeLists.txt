#OpenCOVER plugins general Variant

SET(HEADERS
  coVRBoxOfInterest.h
  Variant.h
  VariantPlugin.h
  VariantUI.h
)

SET(SOURCES
  coVRBoxOfInterest.cpp
  Variant.cpp
  VariantPlugin.cpp
  VariantUI.cpp
)

cover_add_plugin(Variant)
qt_use_modules(Variant Xml)
