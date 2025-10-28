#pragma once
#include <cover/coBillboard.h>
#include <lib/core/interfaces/IInfoboard.h>

#include <osg/ref_ptr>

struct OsgTxtBoxAttributes {
  OsgTxtBoxAttributes(const osg::Vec3 &pos, const std::string &title,
                   const std::string &font, const float &width, const float &height,
                   const float &margin, const float &titleHeightPercentage,
                   int charSize = 2)
      : position(pos),
        title(title),
        fontFile(font),
        maxWidth(width),
        height(height),
        margin(margin),
        titleHeightPercentage(titleHeightPercentage),
        charSize(charSize) {}
  osg::Vec3 position;
  std::string title;
  std::string fontFile;
  float maxWidth;
  float height;
  float margin;
  float titleHeightPercentage;  // title height in percentage of total height
  int charSize;
};

/**
 * @class OsgTxtInfoboard
 * @brief A text-based infoboard for displaying information in an OpenCOVER scene.
 *
 * TxtInfoboard provides an interface for showing, hiding, and updating textual
 * information on a 3D infoboard. It supports customization of appearance via
 * attributes such as position, title, font, size, and margins. The class inherits
 * from core::interface::IInfoboard and implements its required methods for managing
 * the infoboard's lifecycle and content.
 *
 * @note The infoboard uses OSG (OpenSceneGraph) objects for rendering and
 * positioning.
 *
 * @see core::interface::IInfoboard
 */
class OsgTxtInfoboard : public core::interface::IInfoboard<std::string, osg::ref_ptr<osg::Node>> {
 public:
  OsgTxtInfoboard(const OsgTxtBoxAttributes &attributes)
      : m_attributes(attributes),
        m_TextGeode(nullptr),
        m_BBoard(nullptr),
        m_drawable(nullptr),
        m_enabled(false),
        m_info("") {};
  OsgTxtInfoboard(const osg::Vec3 &position, const std::string &title,
               const std::string &font, const float &maxWidth, const float &height,
               const float &margin, const float &titleHeightPercentage,
               int charSize = 2)
      : OsgTxtInfoboard(OsgTxtBoxAttributes(position, title, font, maxWidth, height,
                                      margin, titleHeightPercentage, charSize)) {};

  // IInfoboard interface
  void showInfo() override;
  void hideInfo() override;
  void initDrawable() override;
  void initInfoboard() override;
  void updateDrawable() override;
  void updateInfo(const std::string &info) override;
  void move(const osg::Vec3 &pos) override;
  bool enabled() override { return m_enabled; }

  // getter and setter
  void setMaxWidth(float width) { m_attributes.maxWidth = width; }
  void setHeight(float height) { m_attributes.height = height; }
  void setFont(const std::string &font) { m_attributes.fontFile = font; }
  void setTitle(const std::string &title) { m_attributes.title = title; }
  void setCharSize(int charSize) { m_attributes.charSize = charSize; }

  osg::ref_ptr<osg::Node> &getDrawable() override { return m_drawable; }
  [[nodiscard]] const auto &getMaxWidth(float width) {
    return m_attributes.maxWidth;
  }
  [[nodiscard]] const auto &getHeight(float height) { return m_attributes.height; }
  [[nodiscard]] const auto &getFont(const std::string &font) {
    return m_attributes.fontFile;
  }
  [[nodiscard]] const auto &getTitle(const std::string &title) {
    return m_attributes.title;
  }
  [[nodiscard]] const auto &getCharSize(int charSize) {
    return m_attributes.charSize;
  }
  [[nodiscard]] const auto &getDrawable() const { return m_drawable; }

 private:
  osg::ref_ptr<osg::Group> m_TextGeode;
  osg::ref_ptr<opencover::coBillboard> m_BBoard;
  osg::ref_ptr<osg::Node> m_drawable;

  // txtbox attributes
  OsgTxtBoxAttributes m_attributes;
  bool m_enabled;
  std::string m_info;
};
