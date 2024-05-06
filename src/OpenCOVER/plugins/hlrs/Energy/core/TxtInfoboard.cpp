#include "TxtInfoboard.h"
#include "cover/coBillboard.h"
#include <cover/coVRFileManager.h>

#include <osg/Geode>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osgText/Text>

namespace {
void deleteChildrenRecursive(osg::Group *grp)
{
    if (!grp)
        return;

    for (int i = 0; i < grp->getNumChildren(); ++i) {
        auto child = grp->getChild(i);
        if (auto child_group = dynamic_cast<osg::Group *>(child))
            deleteChildrenRecursive(child_group);
        grp->removeChild(child);
    }
}
} // namespace

namespace core {

void TxtInfoboard::updateTime(int timestep)
{
    //TODO: implement later when needed
}

void TxtInfoboard::initDrawable()
{
    m_drawable = m_BBoard;
}

void TxtInfoboard::initInfoboard()
{
    m_BBoard = new opencover::coBillboard();
    m_BBoard->setNormal(osg::Vec3(0, -1, 0));
    m_BBoard->setAxis(osg::Vec3(0, 0, 1));
    m_BBoard->setMode(opencover::coBillboard::AXIAL_ROT);
}

void TxtInfoboard::updateDrawable()
{}

void TxtInfoboard::updateInfo(const std::string &info)
{
    m_info = info;
    deleteChildrenRecursive(m_BBoard);
    const auto titlePos = m_attributes.position;
    auto contentPos = titlePos;
    contentPos.z() -= m_attributes.height * m_attributes.titleHeightPercentage;
    auto textBoxTitle = createTextBox(m_attributes.title, titlePos, m_attributes.charSize, m_attributes.fontFile.c_str(), m_attributes.maxWidth, m_attributes.margin);
    auto textBoxContent = createTextBox("", contentPos, m_attributes.charSize, m_attributes.fontFile.c_str(), m_attributes.maxWidth, m_attributes.margin);
    textBoxContent->setText(info, osgText::String::ENCODING_UTF8);

    osg::ref_ptr<osg::Geode> geo = new osg::Geode();
    geo->setName("TextBox");
    geo->addDrawable(textBoxTitle);
    geo->addDrawable(textBoxContent);

    m_TextGeode = new osg::Group();
    m_TextGeode->setName("TextGroup");
    m_TextGeode->addChild(geo);
    if (m_enabled)
        showInfo();
}

void TxtInfoboard::showInfo()
{
    m_BBoard->addChild(m_TextGeode);
    m_enabled = true;
}

void TxtInfoboard::move(const osg::Vec3 &pos)
{
    m_attributes.position = pos;
    if (m_enabled)
        updateInfo(m_info);
}

void TxtInfoboard::hideInfo()
{
    m_BBoard->removeChild(m_TextGeode);
    m_enabled = false;
}

osg::ref_ptr<osgText::Text> TxtInfoboard::createTextBox(const std::string &text, const osg::Vec3 &position,
                                                        int charSize, const char *fontFile, const float &maxWidth, const float &margin) const
{
    osg::ref_ptr<osgText::Text> textBox = new osgText::Text();
    textBox->setAlignment(osgText::Text::LEFT_TOP);
    textBox->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBox->setColor(osg::Vec4(1, 1, 1, 1));
    textBox->setText(text, osgText::String::ENCODING_UTF8);
    textBox->setCharacterSize(charSize);
    textBox->setFont(opencover::coVRFileManager::instance()->getFontFile(fontFile));
    textBox->setMaximumWidth(maxWidth);
    textBox->setPosition(position);
    textBox->setDrawMode(osgText::Text::FILLEDBOUNDINGBOX | osgText::Text::TEXT);
    textBox->setBoundingBoxColor(osg::Vec4(0.0f, 0.0f, 0.0f, 0.5f));
    textBox->setBoundingBoxMargin(margin);
    return textBox;
}
} // namespace core