// std
#include <iostream>
// ours
#include "Compression.h"
#include "MiniRR.h"

#define LOG_ERR std::cerr

namespace minirr {

struct RenderState
{
  struct {
    int value{0};
    bool updated{false};
  } numChannels;

  struct {
    int width{1};
    int height{1};
    bool updated{false};
  } size;

  struct {
    Mat4 modelMatrix;
    Mat4 viewMatrix;
    Mat4 projMatrix;
    bool updated{false};
  } camera;

  struct {
    AABB aabb;
    bool updated{false};
  } bounds;

  struct {
    std::vector<char> colorBuffer;
    std::vector<char> depthBuffer;
    bool updated{false};
  } frame;

  struct {
    std::vector<uint8_t> data;
    int width{0};
    int height{0};
    uint32_t compressedSize{0};
    bool updated{false};
  } image;
};

MiniRR::MiniRR() : renderState(new RenderState)
{
}

MiniRR::~MiniRR() {}

void MiniRR::initAsClient(std::string hostname, unsigned short port)
{
  mode = Client;
  manager = async::make_connection_manager();
  manager->connect(hostname, port,
      std::bind(&MiniRR::handleNewConnection,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

void MiniRR::initAsServer(unsigned short port)
{
  mode = Server;
  manager = async::make_connection_manager(port);
  manager->accept(std::bind(&MiniRR::handleNewConnection,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

void MiniRR::run()
{
  manager->run_in_thread();
  queue.run_in_thread();

  //if (mode == Server) {
  //  manager->wait();
  //}
  
  std::unique_lock<std::mutex> l(sync[SyncPoints::ConnectionEstablished].mtx);
  sync[SyncPoints::ConnectionEstablished].cv.wait(
      l, [this]() { return conn; });
  l.unlock();
}

bool MiniRR::connectionClosed()
{
  return false; // TODO!
}

void MiniRR::sendNumChannels(int numChannels)
{
  lock();
  renderState->numChannels.value = numChannels;
  renderState->numChannels.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write(renderState->numChannels.value);
  write(MessageType::SendNumChannels, buf);
}

void MiniRR::recvNumChannels(int &numChannels)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvNumChannels, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvNumChannels].mtx);
  sync[SyncPoints::RecvNumChannels].cv.wait(
      l, [this]() { return renderState->numChannels.updated; });
  l.unlock();

  lock();
  numChannels = renderState->numChannels.value;
  renderState->numChannels.updated = false;
  unlock();
}

void MiniRR::sendBounds(AABB bounds)
{
  lock();
  std::memcpy(renderState->bounds.aabb, bounds, sizeof(renderState->bounds.aabb));
  renderState->bounds.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)renderState->bounds.aabb, sizeof(renderState->bounds.aabb));
  write(MessageType::SendBounds, buf);
}

void MiniRR::recvBounds(AABB &bounds)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvBounds, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvBounds].mtx);
  sync[SyncPoints::RecvBounds].cv.wait(
      l, [this]() { return renderState->bounds.updated; });
  l.unlock();

  lock();
  std::memcpy(bounds, renderState->bounds.aabb, sizeof(renderState->bounds.aabb));
  renderState->bounds.updated = false;
  unlock();
}

void MiniRR::sendSize(int w, int h)
{
  lock();
  renderState->size.width = w;
  renderState->size.height = h;
  renderState->size.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write(renderState->size.width);
  buf->write(renderState->size.height);
  write(MessageType::SendSize, buf);
}

void MiniRR::recvSize(int &w, int &h)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvSize, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvSize].mtx);
  sync[SyncPoints::RecvSize].cv.wait(
      l, [this]() { return renderState->size.updated; });
  l.unlock();

  lock();
  w = renderState->size.width;
  h = renderState->size.height;
  renderState->size.updated = false;
  unlock();
}

void MiniRR::sendCamera(Mat4 modelMatrix, Mat4 viewMatrix, Mat4 projMatrix)
{
  lock();
  std::memcpy(renderState->camera.modelMatrix, modelMatrix, sizeof(renderState->camera.modelMatrix));
  std::memcpy(renderState->camera.viewMatrix, viewMatrix, sizeof(renderState->camera.viewMatrix));
  std::memcpy(renderState->camera.projMatrix, projMatrix, sizeof(renderState->camera.projMatrix));
  renderState->camera.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)renderState->camera.modelMatrix, sizeof(renderState->camera.modelMatrix));
  buf->write((const char *)renderState->camera.viewMatrix, sizeof(renderState->camera.viewMatrix));
  buf->write((const char *)renderState->camera.projMatrix, sizeof(renderState->camera.projMatrix));
  write(MessageType::SendCamera, buf);
}

void MiniRR::recvCamera(Mat4 &modelMatrix, Mat4 &viewMatrix, Mat4 &projMatrix)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvCamera, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvCamera].mtx);
  sync[SyncPoints::RecvCamera].cv.wait(
      l, [this]() { return renderState->camera.updated; });
  l.unlock();

  lock();
  std::memcpy(modelMatrix, renderState->camera.modelMatrix, sizeof(renderState->camera.modelMatrix));
  std::memcpy(viewMatrix, renderState->camera.viewMatrix, sizeof(renderState->camera.viewMatrix));
  std::memcpy(projMatrix, renderState->camera.projMatrix, sizeof(renderState->camera.projMatrix));
  renderState->camera.updated = false;
  unlock();
}

void MiniRR::sendImage(const uint32_t *img, int width, int height)
{
  lock();

  TurboJPEGOptions options;
  options.width = width;
  options.height = height;
  options.pixelFormat = TurboJPEGOptions::PixelFormat::RGBX;
  options.quality = 80;

  renderState->image.data.resize(getMaxCompressedBufferSizeTurboJPEG(options));

  size_t compressedSize;
  compressTurboJPEG((const uint8_t *)img,
      renderState->image.data.data(),
      compressedSize,
      options);

  renderState->image.width = width;
  renderState->image.height = height;
  renderState->image.compressedSize = compressedSize;
  renderState->image.updated = true;

  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&renderState->image.width, sizeof(renderState->image.width));
  buf->write((const char *)&renderState->image.height, sizeof(renderState->image.height));
  buf->write((const char *)&renderState->image.compressedSize, sizeof(renderState->image.compressedSize));
  buf->write((const char *)renderState->image.data.data(),
      sizeof(renderState->image.data[0])*renderState->image.data.size());
  write(MessageType::SendImage, buf);
}

void MiniRR::recvImage(uint32_t *img, int &width, int &height)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvImage, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvImage].mtx);
  sync[SyncPoints::RecvImage].cv.wait(
      l, [this]() { return renderState->image.updated; });
  l.unlock();

  lock();

  width = renderState->image.width;
  height = renderState->image.height;

  TurboJPEGOptions options;
  options.width = width;
  options.height = height;
  options.pixelFormat = TurboJPEGOptions::PixelFormat::RGBX;
  options.quality = 80;

  uncompressTurboJPEG(renderState->image.data.data(),
      (uint8_t *)img,
      (size_t)renderState->image.compressedSize,
      options);

  renderState->image.updated = false;

  unlock();
}

bool MiniRR::handleNewConnection(
    async::connection_pointer new_conn, const boost::system::error_code &e)
{
  if (e) {
    LOG_ERR << "connection failed";
    manager->stop();
    return false;
  }

  conn = new_conn;
  conn->set_handler(std::bind(&MiniRR::handleMessage,
    this,
    std::placeholders::_1,
    std::placeholders::_2,
    std::placeholders::_3));

  sync[SyncPoints::ConnectionEstablished].cv.notify_all();

  // wait for new connections (these will overwrite the current one though!!)
  //if (mode == Server) {
  //  manager->accept(std::bind(&MiniRR::handleNewConnection,
  //    this,
  //    std::placeholders::_1,
  //    std::placeholders::_2));
  //}

  return true;
}

void MiniRR::handleMessage(async::connection::reason reason,
    async::message_pointer message,
    const boost::system::error_code &e)
{
  if (e) {
    LOG_ERR << "error handling message";
    manager->stop();
    return;
  }

  if (reason == async::connection::Read) {
    handleReadMessage(message);
  }

  if (reason == async::connection::Write) {
    handleWriteMessage(message);
  }
}

void MiniRR::handleReadMessage(async::message_pointer message)
{
  auto buf = std::make_shared<Buffer>(message->data(), message->size());
  if (message->type() == MessageType::SendNumChannels) {
    lock();
    buf->read(renderState->numChannels.value);
    renderState->numChannels.updated = true;
    unlock();
    sync[SyncPoints::RecvNumChannels].cv.notify_all();
  }
  else if (message->type() == MessageType::SendBounds) {
    lock();
    buf->read((char *)renderState->bounds.aabb, sizeof(renderState->bounds.aabb));
    renderState->bounds.updated = true;
    sync[SyncPoints::RecvBounds].cv.notify_all();
    unlock();
    sync[SyncPoints::RecvBounds].cv.notify_all();
  }
  else if (message->type() == MessageType::SendSize) {
    lock();
    buf->read(renderState->size.width);
    buf->read(renderState->size.height);
    renderState->size.updated = true;
    unlock();
    sync[SyncPoints::RecvSize].cv.notify_all();
  }
  else if (message->type() == MessageType::SendCamera) {
    lock();
    buf->read((char *)renderState->camera.modelMatrix, sizeof(renderState->camera.modelMatrix));
    buf->read((char *)renderState->camera.viewMatrix, sizeof(renderState->camera.viewMatrix));
    buf->read((char *)renderState->camera.projMatrix, sizeof(renderState->camera.projMatrix));
    renderState->camera.updated = true;
    unlock();
    sync[SyncPoints::RecvCamera].cv.notify_all();
  }
  else if (message->type() == MessageType::SendImage) {
    lock();
    buf->read((char *)&renderState->image.width, sizeof(renderState->image.width));
    buf->read((char *)&renderState->image.height, sizeof(renderState->image.height));
    buf->read((char *)&renderState->image.compressedSize, sizeof(renderState->image.compressedSize));
    renderState->image.data.resize(renderState->image.compressedSize);
    buf->read((char *)renderState->image.data.data(), renderState->image.compressedSize);
    renderState->image.updated = true;
    unlock();
    sync[SyncPoints::RecvImage].cv.notify_all();
  }
}

void MiniRR::handleWriteMessage(async::message_pointer message)
{
  // if (message->type() == MessageType::SendNumChannels) {
  //   sync[SyncPoints::SendNumChannels].cv.notify_all();
  // }
  // else if (message->type() == MessageType::SendSize) {
  //   sync[SyncPoints::SendSize].cv.notify_all();
  // }
  // else if (message->type() == MessageType::SendCamera) {
  //   sync[SyncPoints::SendCamera].cv.notify_all();
  // }
  // else if (message->type() == MessageType::SendBounds) {
  //   sync[SyncPoints::SendBounds].cv.notify_all();
  // }
}

void MiniRR::write(unsigned type, std::shared_ptr<Buffer> buf)
{
  queue.post(std::bind(&MiniRR::writeImpl, this, type, buf));
}

void MiniRR::write(unsigned type, const void *begin, const void *end)
{
  queue.post(std::bind(&MiniRR::writeImpl2, this, type, begin, end));
}

void MiniRR::writeImpl(unsigned type, std::shared_ptr<Buffer> buf)
{
  conn->write(type, *buf);
}

void MiniRR::writeImpl2(unsigned type, const void *begin, const void *end)
{
  conn->write(type, (const char *)begin, (const char *)end);
}

} // namespace minirr

#define TEST

#ifdef TEST
int main(int argc, char **argv) {
  if (argc == 1) {
    std::cerr << "minirrTest [client|server]\n";
    exit(-1);
  }
  if (std::string(argv[1]) == "client") {
    minirr::MiniRR rr;
    rr.initAsClient("localhost", 31050);
    rr.run();
    while (1) {
      rr.sendSize(1024,768);
    }
  } else {
    minirr::MiniRR rr;
    rr.initAsServer(31050);
    rr.run();
  }
}
#endif
