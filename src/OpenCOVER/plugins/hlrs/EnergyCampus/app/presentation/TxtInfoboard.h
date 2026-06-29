#pragma once
#include <cover/coBillboard.h>
#include <lib/core/interfaces/IInfoboard.h>

#include <osg/ref_ptr>

struct TxtBoxAttributes {
  TxtBoxAttributes(const osg::Vec3 &pos, const std::string &title,
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
 * @class TxtInfoboard
 * @brief A text-based infoboard for displaying information in an OpenCOVER scene.
 *
 * TxtInfoboard provides an interface for showing, hiding, and updating textual information
 * on a 3D infoboard. It supports customization of appearance via attributes such as position,
 * title, font, size, and margins. The class inherits from core::interface::IInfoboard and
 * implements its required methods for managing the infoboard's lifecycle and content.
 *
 * @note The infoboard uses OSG (OpenSceneGraph) objects for rendering and positioning.
 *
 * @see core::interface::IInfoboard
 */
class TxtInfoboard : public core::interface::IInfoboard<std::string> {
 public:
  TxtInfoboard(const TxtBoxAttributes &attributes) : m_attributes(attributes) {};
  TxtInfoboard(const osg::Vec3 &position, const std::string &title,
               const std::string &font, const float &maxWidth, const float &height,
               const float &margin, const float &titleHeightPercentage,
               int charSize = 2)
      : m_attributes(TxtBoxAttributes(position, title, font, maxWidth, height,
                                      margin, titleHeightPercentage, charSize)) {};

  // IInfoboard interface
  void updateTime(int timestep) override;
  void showInfo() override;
  void hideInfo() override;
  void initDrawable() override;
  void initInfoboard() override;
  void updateDrawable() override;
  void updateInfo(const std::string &info) override;
  void move(const osg::Vec3 &pos) override;

  // getter and setter
  void setMaxWidth(float width) { m_attributes.maxWidth = width; }
  void setHeight(float height) { m_attributes.height = height; }
  void setFont(const std::string &font) { m_attributes.fontFile = font; }
  void setTitle(const std::string &title) { m_attributes.title = title; }
  void setCharSize(int charSize) { m_attributes.charSize = charSize; }
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

 private:
  osg::ref_ptr<osg::Group> m_TextGeode = nullptr;
  osg::ref_ptr<opencover::coBillboard> m_BBoard = nullptr;

  // txtbox attributes
  TxtBoxAttributes m_attributes;
};
