#include "EnergyGrid.h"

#include <lib/core/constants.h>
#include <lib/core/simulation/grid.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

#include <cassert>
// #include <osg/MatrixTransform>
// #include <osg/PositionAttitudeTransform>
#include <cstddef>
#include <osg/Shape>
// #include <osg/Transform>
#include <osg/ref_ptr>
#include <osgText/Text>
// #include <osgViewer/Viewer>
#include <sstream>
#include <variant>

#include "TxtInfoboard.h"
#include "cover/VRViewer.h"

namespace {
auto get_string = [](const auto &data) {
  std::stringstream ss;
  ss << data << "\n\n";
  return ss.str();
};
}  // namespace
// #include "cover/coBillboard.h"

// namespace {
// class BillboardTextCallback : public osg::NodeCallback {
//  public:
//   BillboardTextCallback(osg::Camera *camera) : _camera(camera) {}

//   void operator()(osg::Node *node, osg::NodeVisitor *nv) {
//     // Get the PositionAttitudeTransform
//     osg::PositionAttitudeTransform *pat =
//         dynamic_cast<osg::PositionAttitudeTransform *>(node);
//     if (pat) {
//       // Update the position of the text in front of the camera
//       osg::Vec3 cameraPosition = _camera->getViewMatrix().getTrans();
//       osg::Vec3 eye, center, up;
//       _camera->getViewMatrixAsLookAt(eye, center, up);
//       osg::Vec3 cameraDirection = center - eye;
//       pat->setPosition(cameraPosition +
//                        cameraDirection * 5.0f);  // Adjust the distance as needed

//       // Update the orientation of the text to face the camera
//       pat->setAttitude(osg::Quat());  // Reset attitude to face the camera
//     }

//     traverse(node, nv);
//   }

//  private:
//   osg::ref_ptr<osg::Camera> _camera;
// };
// }  // namespace

EnergyGrid::InfoboardSensor::InfoboardSensor(
    osg::ref_ptr<osg::Group> parent,
    std::unique_ptr<interface::IInfoboard<std::string>> &&infoboard,
    const std::string &content)
    : coPickSensor(parent), m_infoBoard(std::move(infoboard)) {
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  m_infoBoard->updateInfo(content);
  parent->addChild(m_infoBoard->getDrawable());
}

int EnergyGrid::InfoboardSensor::hit(vrui::vruiHit *hit) {
  if (!interaction) return 0;

  // click on object and hold to show info
  if (interaction->wasStarted() && !active) {
    m_infoBoard->showInfo();
    active = true;
  }
  // release to hide info if your not out of the object
  // otherwise the billboard will be shown until you click again
  if (interaction->wasStopped() && active) {
    m_infoBoard->hideInfo();
    active = false;
  }
  return coPickSensor::hit(hit);
}

EnergyGrid::EnergyGrid(EnergyGridConfig &&data) : m_config(std::move(data)) {
  if (m_config.parent == nullptr) {
    m_config.parent = new osg::Group;
    m_config.parent->setName(m_config.name);
  }
  initConnections(m_config.indices, m_config.connectionRadius,
                  m_config.additionalConnectionData);
};

void EnergyGrid::initConnections(const grid::Indices &indices, const float &radius,
                                 const grid::DataList &additionalConnectionData) {
  bool hasAdditionalData = !additionalConnectionData.empty();

  const auto &points = m_config.points;
  for (auto i = 0; i < indices.size(); ++i) {
    for (auto j = 0; j < indices[i].size(); ++j) {
      std::unique_ptr<grid::ConnectionData<grid::Point>> data;

      const auto indice = indices[i][j];
      if (i < 0 || i >= points.size()) {
        std::cerr << "Invalid Index for points: " << i << "\n";
        continue;
      }
      const auto &from = *points[i];

      if (indice >= points.size() || indice < 0) {
        std::cerr << "Invalid Index for points: " << indice << "\n";
        continue;
      }
      const auto &to = *points[indice];

      std::string name(from.getName() + " " + UIConstants::RIGHT_ARROW_UNICODE_HEX +
                       " " + to.getName());
      core::simulation::grid::Data additionalData{};
      if (hasAdditionalData)
        if (additionalConnectionData.size() > i + j)
          additionalData = additionalConnectionData[i + j];
      data = std::make_unique<grid::ConnectionData<grid::Point>>(
          name, from, to, radius, nullptr, additionalData);
      m_connections.push_back(new grid::DirectedConnection(*data));
    }
  }
}

void EnergyGrid::initDrawablePoints() {
  osg::ref_ptr<osg::Group> points = new osg::Group;
  points->setName("Points");
  for (auto &point : m_config.points) {
    m_drawables.push_back(point);
    points->addChild(point);
    std::string toPrint = "";
    for (const auto &[name, data] : point->getAdditionalData()) {
      toPrint +=
          UIConstants::TAB_SPACES + name + ": " + std::visit(get_string, data);
    }
    auto center = point->getPosition();
    auto pointBB = point->getGeode()->getBoundingBox();
    center.z() += 30;
    auto name = point->getName();

    m_config.infoboardAttributes.position = center;
    m_config.infoboardAttributes.title = name;
    TxtInfoboard infoboard(m_config.infoboardAttributes);
    m_infoboards.push_back(std::make_unique<InfoboardSensor>(
        point, std::make_unique<TxtInfoboard>(infoboard), toPrint));
  }
  m_config.parent->addChild(points);
}

osg::ref_ptr<grid::DirectedConnection> EnergyGrid::getConnectionByName(
    const std::string &name) {
  for (auto &connection : m_connections)
    if (connection->getName() == name) return connection;
  return nullptr;
}

osg::ref_ptr<grid::Point> EnergyGrid::getPointByName(const std::string &name) {
  for (auto &point : m_config.points)
    if (point->getName() == name) return point;
  return nullptr;
}

void EnergyGrid::initDrawableConnections() {
  osg::ref_ptr<osg::Group> connections = new osg::Group;
  connections->setName("Connections");

  //   auto viewer = opencover::VRViewer::instance();
  //   auto camera = viewer->getCamera();
  for (auto &connection : m_connections) {
    m_drawables.push_back(connection);
    connections->addChild(connection);

    std::string toPrint = "";
    for (const auto &[name, data] : connection->getAdditionalData()) {
      toPrint +=
          UIConstants::TAB_SPACES + name + ": " + std::visit(get_string, data);
    }
    auto center = connection->getCenter();
    auto connectionBB = connection->getGeode()->getBoundingBox();
    center.z() += 30;
    auto name = connection->getName();

    m_config.infoboardAttributes.position = center;
    m_config.infoboardAttributes.title = name;
    TxtInfoboard infoboard(m_config.infoboardAttributes);
    m_infoboards.push_back(std::make_unique<InfoboardSensor>(
        connection, std::make_unique<TxtInfoboard>(infoboard), toPrint));

    // TODO: use the position of the viewer to determine the position of the
    // infoboard
    // auto iboard = m_infoboards.back()->getInfoboard();
    // iboard->showInfo(); //otherwise the infoboard would have geode attached
    // auto mt = dynamic_cast<osg::MatrixTransform *>(iboard->getDrawable().get());
    // if (!mt)
    //     std::cout << "Error: Could not get MatrixTransform from infoboard
    //     drawable\n";
    // auto bboard = dynamic_cast<osg::Transform *>(mt->getChild(0));
    // if (!bboard)
    //     std::cout << "Error: Could not get Billboard from infoboard drawable\n";
    // auto geo = bboard->getChild(0)->asGeode();
    // if (!geo)
    //     std::cout << "Error: Could not get Geode from infoboard drawable\n";

    // auto geo =
    // iboard->getDrawable()->asMatrixTransform()->getChild(0)->asGeode(); if
    // (!geo) {
    //     std::cout << "Error: Could not get Geode from infoboard drawable\n";
    //     return;
    // }

    // auto parent = dynamic_cast<osg::MatrixTransform *>(geo->getParent(0));
    // osg::ref_ptr<osg::PositionAttitudeTransform> pat = new
    // osg::PositionAttitudeTransform; mt->addChild(pat);

    // pat->addChild(geo);
    // mt->removeChild(geo);

    // mt->setUpdateCallback(new BillboardTextCallback(camera));
    // iboard->getDrawable()->setUpdateCallback(new BillboardTextCallback(camera));
  }

  m_config.parent->addChild(connections);
}

void EnergyGrid::initDrawables() {
  initDrawablePoints();
  initDrawableConnections();
}

void EnergyGrid::updateColor(const osg::Vec4 &color) {
  for (auto &connection : m_connections) {
    utils::color::overrideGeodeColor(connection->getGeode(), color);
  }
  for (auto &point : m_config.points) {
    utils::color::overrideGeodeColor(point->getGeode(), color);
  }
}

void EnergyGrid::updateDrawables() {
  for (auto &infoboard : m_infoboards) {
    infoboard->updateDrawable();
  }
}
