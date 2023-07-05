#ifndef COVERPLUGINANIMATED_AVATAR_H
#define COVERPLUGINANIMATED_AVATAR_H

#include <osgCal/CoreModel>//abstraktes Modell
#include <osgCal/Model>//konkretes Modell in der Szene

#include <cover/coVRPartner.h>

class AnimatedAvatar
{
public:
  AnimatedAvatar(const osg::ref_ptr<osgCal::CoreModel> &model, int partnerId); //initialisiert das entsprechende model aus dem core model und setzt es auf die position des partners (mittels m_transform) und hängt es in den scenengraph
  //da wir einen destruktor brauchen empfielt es sich die folgenden copy und move operationen zu implementiren
  //copy wird verboten und macht this zu other und other kaputt 
  AnimatedAvatar(const AnimatedAvatar &other) = delete;
  AnimatedAvatar(AnimatedAvatar &&other) = default;
  AnimatedAvatar& operator=(const AnimatedAvatar &other) = delete;
  AnimatedAvatar& operator=(AnimatedAvatar &&other) = default;

  ~AnimatedAvatar(); //löscht das model aus dem scenengraph 
  void update(); //checkt die bewegung des partners (bei coVRPartnerList) und macht die entspechenden animationen
private:
  osg::ref_ptr<osg::MatrixTransform>m_transform;//Position des Avatars
  osgCal::Model *m_model = nullptr;
  int m_partnerId = -1;
  osg::Matrix m_lastHeadPosition; //letzte position um die bewegung fest zu stellen
  osg::Matrix m_lastHandPosition; //letzte position um die bewegung fest zu stellen
  opencover::coVRPartner *m_partner = nullptr;
  void setExclusiveAnimation(int animationsIndex);
};

#endif // COVERPLUGINANIMATED_AVATAR_H