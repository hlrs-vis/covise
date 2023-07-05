#ifndef COVERPLUGINANIMATED_AVATAR_H
#define COVERPLUGINANIMATED_AVATAR_H

#include <osgCal/CoreModel>//abstraktes Modell
#include <osgCal/Model>//konkretes Modell in der Szene
#include <queue>

#include <cover/coVRPartner.h>
#include "ModelProvider/ModelProvider.h"

class AnimatedAvatar
{
public:
  AnimatedAvatar(const std::string &filename, int partnerId); //initialisiert das entsprechende model aus dem core model und setzt es auf die position des partners (mittels m_transform) und hängt es in den scenengraph
  //da wir einen destruktor brauchen empfielt es sich die folgenden copy und move operationen zu implementiren
  //copy wird verboten und macht this zu other und other kaputt 
  AnimatedAvatar(const AnimatedAvatar &other) = delete;
  AnimatedAvatar(AnimatedAvatar &&other) = default;
  AnimatedAvatar& operator=(const AnimatedAvatar &other) = delete;
  AnimatedAvatar& operator=(AnimatedAvatar &&other) = default;

  void update(); //checkt die bewegung des partners (bei coVRPartnerList) und macht die entspechenden animationen
private:
  std::unique_ptr<ModelProvider> m_model;
  int m_partnerId = -1;
  std::queue<osg::Matrix> m_lastHeadPositions; //letzte position um die bewegung fest zu stellen
  std::queue<osg::Matrix> m_lastHandPositions; //letzte position um die bewegung fest zu stellen
  opencover::coVRPartner *m_partner = nullptr;
};

#endif // COVERPLUGINANIMATED_AVATAR_H
