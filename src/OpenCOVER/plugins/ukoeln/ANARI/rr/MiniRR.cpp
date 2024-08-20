// std
#include <iostream>
// ours
#include "Compression.h"
#include "MiniRR.h"

#ifdef __GNUC__
#include <execinfo.h>
#include <sys/time.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

#define LOG_ERR std::cerr

// #define TIMING

inline double getCurrentTime()
{
#ifdef _WIN32
  SYSTEMTIME tp; GetSystemTime(&tp);
  /*
     Please note: we are not handling the "leap year" issue.
 */
  size_t numSecsSince2020
      = tp.wSecond
      + (60ull) * tp.wMinute
      + (60ull * 60ull) * tp.wHour
      + (60ull * 60ul * 24ull) * tp.wDay
      + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
  return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
  struct timeval tp; gettimeofday(&tp,nullptr);
  return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
}

namespace minirr {

struct State
{
  struct {
    int value{0};
    bool updated{false};
  } numChannels;

  struct {
    Viewport value;
    bool updated{false};
  } viewport;

  struct {
    Camera value;
    bool updated{false};
  } camera;

  struct {
    uint64_t value{0};
    bool updated{false};
  } objectUpdates;

  struct {
    AABB aabb;
    bool updated{false};
  } bounds;

  struct {
    std::vector<float> rgb;
    std::vector<float> alpha;
    uint32_t numRGB{0};
    uint32_t numAlpha{0};
    float absRange[2]{0.f, 1.f};
    float relRange[2]{0.f, 1.f};
    float opacityScale{1.f};
    bool updated{false};
  } transfunc;

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

MiniRR::MiniRR() : sendState(new State), recvState(new State)
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

void MiniRR::sendNumChannels(const int &numChannels)
{
  lock();
  sendState->numChannels.value = numChannels;
  sendState->numChannels.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write(sendState->numChannels.value);
  write(MessageType::SendNumChannels, buf);
}

void MiniRR::recvNumChannels(int &numChannels)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvNumChannels, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvNumChannels].mtx);
  sync[SyncPoints::RecvNumChannels].cv.wait(
      l, [this]() { return recvState->numChannels.updated; });
  l.unlock();

  lock();
  numChannels = recvState->numChannels.value;
  recvState->numChannels.updated = false;
  unlock();
}

void MiniRR::sendViewport(const Viewport &viewport)
{
  lock();
  sendState->viewport.value = viewport;
  sendState->viewport.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&sendState->viewport.value, sizeof(sendState->viewport.value));
  write(MessageType::SendViewport, buf);
}

void MiniRR::recvViewport(Viewport &viewport)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvViewport, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvViewport].mtx);
  sync[SyncPoints::RecvViewport].cv.wait(
      l, [this]() { return recvState->viewport.updated; });
  l.unlock();

  lock();
  viewport = recvState->viewport.value;
  recvState->viewport.updated = false;
  unlock();
}

void MiniRR::sendCamera(const Camera &camera)
{
  lock();
  sendState->camera.value = camera;
  sendState->camera.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&sendState->camera.value, sizeof(sendState->camera.value));
  write(MessageType::SendCamera, buf);
}

void MiniRR::recvCamera(Camera &camera)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvCamera, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvCamera].mtx);
  sync[SyncPoints::RecvCamera].cv.wait(
      l, [this]() { return recvState->camera.updated; });
  l.unlock();

  lock();
  camera = recvState->camera.value;
  recvState->camera.updated = false;
  unlock();
}

void MiniRR::sendObjectUpdates(const uint64_t &objectUpdates)
{
  lock();
  sendState->objectUpdates.value = objectUpdates;
  sendState->objectUpdates.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write(sendState->objectUpdates.value);
  write(MessageType::SendObjectUpdates, buf);
}

void MiniRR::recvObjectUpdates(uint64_t &objectUpdates)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvObjectUpdates, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvObjectUpdates].mtx);
  sync[SyncPoints::RecvObjectUpdates].cv.wait(
      l, [this]() { return recvState->objectUpdates.updated; });
  l.unlock();

  lock();
  objectUpdates = recvState->objectUpdates.value;
  recvState->objectUpdates.updated = false;
  unlock();
}

void MiniRR::sendBounds(const AABB &bounds)
{
  lock();
  std::memcpy(sendState->bounds.aabb, bounds, sizeof(sendState->bounds.aabb));
  sendState->bounds.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)sendState->bounds.aabb, sizeof(sendState->bounds.aabb));
  write(MessageType::SendBounds, buf);
}

void MiniRR::recvBounds(AABB &bounds)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvBounds, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvBounds].mtx);
  sync[SyncPoints::RecvBounds].cv.wait(
      l, [this]() { return recvState->bounds.updated; });
  l.unlock();

  lock();
  std::memcpy(bounds, recvState->bounds.aabb, sizeof(recvState->bounds.aabb));
  recvState->bounds.updated = false;
  unlock();
}

void MiniRR::sendTransfunc(const Transfunc &transfunc)
{
  lock();
  sendState->transfunc.rgb.resize(transfunc.numRGB*3);
  std::memcpy(sendState->transfunc.rgb.data(),
              transfunc.rgb,
              sizeof(transfunc.rgb[0])*transfunc.numRGB*3);
  sendState->transfunc.alpha.resize(transfunc.numAlpha);
  std::memcpy(sendState->transfunc.alpha.data(),
              transfunc.alpha,
              sizeof(transfunc.alpha[0])*transfunc.numAlpha);
  sendState->transfunc.numRGB = transfunc.numRGB;
  sendState->transfunc.numAlpha = transfunc.numAlpha;
  std::memcpy(sendState->transfunc.absRange,
              transfunc.absRange, sizeof(transfunc.absRange));
  std::memcpy(sendState->transfunc.relRange,
              transfunc.relRange, sizeof(transfunc.relRange));
  sendState->transfunc.opacityScale = transfunc.opacityScale;
  sendState->transfunc.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&sendState->transfunc.numRGB,
      sizeof(sendState->transfunc.numRGB));
  buf->write((const char *)&sendState->transfunc.numAlpha,
      sizeof(sendState->transfunc.numAlpha));
  buf->write((const char *)&sendState->transfunc.absRange,
      sizeof(sendState->transfunc.absRange));
  buf->write((const char *)&sendState->transfunc.relRange,
      sizeof(sendState->transfunc.relRange));
  buf->write((const char *)&sendState->transfunc.opacityScale,
      sizeof(sendState->transfunc.opacityScale));
  buf->write((const char *)sendState->transfunc.rgb.data(),
      sizeof(sendState->transfunc.rgb[0])*sendState->transfunc.rgb.size());
  buf->write((const char *)sendState->transfunc.alpha.data(),
      sizeof(sendState->transfunc.alpha[0])*sendState->transfunc.alpha.size());
  write(MessageType::SendTransfunc, buf);
}

void MiniRR::recvTransfunc(Transfunc &transfunc)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvTransfunc, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvTransfunc].mtx);
  sync[SyncPoints::RecvTransfunc].cv.wait(
      l, [this]() { return recvState->transfunc.updated; });
  l.unlock();

  lock();

  transfunc.rgb = recvState->transfunc.rgb.data();
  transfunc.alpha = recvState->transfunc.alpha.data();
  transfunc.numRGB = recvState->transfunc.numRGB;
  transfunc.numAlpha = recvState->transfunc.numAlpha;
  transfunc.absRange[0] = recvState->transfunc.absRange[0];
  transfunc.absRange[1] = recvState->transfunc.absRange[1];
  transfunc.relRange[0] = recvState->transfunc.relRange[0];
  transfunc.relRange[1] = recvState->transfunc.relRange[1];
  transfunc.opacityScale = recvState->transfunc.opacityScale;

  recvState->transfunc.updated = false;

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

  sendState->image.data.resize(getMaxCompressedBufferSizeTurboJPEG(options));

  #ifdef TIMING
  double t0 = getCurrentTime();
  #endif
  size_t compressedSize;
  compressTurboJPEG((const uint8_t *)img,
      sendState->image.data.data(),
      compressedSize,
      options);
  #ifdef TIMING
  double t1 = getCurrentTime();
  std::cout << "compressTurboJPEG takes " << (t1-t0) << " sec.\n";
  #endif

  sendState->image.width = width;
  sendState->image.height = height;
  sendState->image.compressedSize = compressedSize;
  sendState->image.updated = true;

  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&sendState->image.width, sizeof(sendState->image.width));
  buf->write((const char *)&sendState->image.height, sizeof(sendState->image.height));
  buf->write((const char *)&sendState->image.compressedSize, sizeof(sendState->image.compressedSize));
  buf->write((const char *)sendState->image.data.data(),
      sizeof(sendState->image.data[0])*sendState->image.data.size());
  write(MessageType::SendImage, buf);
}

void MiniRR::recvImage(uint32_t *img, int &width, int &height)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvImage, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvImage].mtx);
  sync[SyncPoints::RecvImage].cv.wait(
      l, [this]() { return recvState->image.updated; });
  l.unlock();

  lock();

  width = recvState->image.width;
  height = recvState->image.height;

  TurboJPEGOptions options;
  options.width = width;
  options.height = height;
  options.pixelFormat = TurboJPEGOptions::PixelFormat::RGBX;
  options.quality = 80;

  #ifdef TIMING
  double t0 = getCurrentTime();
  #endif
  uncompressTurboJPEG(recvState->image.data.data(),
      (uint8_t *)img,
      (size_t)recvState->image.compressedSize,
      options);
  #ifdef TIMING
  double t1 = getCurrentTime();
  std::cout << "uncompressTurboJPEG takes " << (t1-t0) << " sec.\n";
  #endif

  recvState->image.updated = false;

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
    buf->read(recvState->numChannels.value);
    recvState->numChannels.updated = true;
    unlock();
    sync[SyncPoints::RecvNumChannels].cv.notify_all();
  }
  else if (message->type() == MessageType::SendViewport) {
    lock();
    buf->read((char *)&recvState->viewport.value, sizeof(recvState->viewport.value));
    recvState->viewport.updated = true;
    unlock();
    sync[SyncPoints::RecvViewport].cv.notify_all();
  }
  else if (message->type() == MessageType::SendCamera) {
    lock();
    buf->read((char *)&recvState->camera.value, sizeof(recvState->camera.value));
    recvState->camera.updated = true;
    unlock();
    sync[SyncPoints::RecvCamera].cv.notify_all();
  }
  if (message->type() == MessageType::SendObjectUpdates) {
    lock();
    buf->read(recvState->objectUpdates.value);
    recvState->objectUpdates.updated = true;
    unlock();
    sync[SyncPoints::RecvObjectUpdates].cv.notify_all();
  }
  else if (message->type() == MessageType::SendBounds) {
    lock();
    buf->read((char *)recvState->bounds.aabb, sizeof(recvState->bounds.aabb));
    recvState->bounds.updated = true;
    unlock();
    sync[SyncPoints::RecvBounds].cv.notify_all();
  }
  else if (message->type() == MessageType::SendTransfunc) {
    lock();
    buf->read((char *)&recvState->transfunc.numRGB, sizeof(recvState->transfunc.numRGB));
    buf->read((char *)&recvState->transfunc.numAlpha, sizeof(recvState->transfunc.numAlpha));
    buf->read((char *)&recvState->transfunc.absRange, sizeof(recvState->transfunc.absRange));
    buf->read((char *)&recvState->transfunc.relRange, sizeof(recvState->transfunc.relRange));
    buf->read((char *)&recvState->transfunc.opacityScale, sizeof(recvState->transfunc.opacityScale));
    recvState->transfunc.rgb.resize(recvState->transfunc.numRGB*3);
    buf->read((char *)recvState->transfunc.rgb.data(),
        sizeof(recvState->transfunc.rgb[0])*recvState->transfunc.rgb.size());
    recvState->transfunc.alpha.resize(recvState->transfunc.numAlpha);
    buf->read((char *)recvState->transfunc.alpha.data(),
        sizeof(recvState->transfunc.alpha[0])*recvState->transfunc.alpha.size());
    recvState->transfunc.updated = true;
    unlock();
    sync[SyncPoints::RecvTransfunc].cv.notify_all();
  }
  else if (message->type() == MessageType::SendImage) {
    lock();
    buf->read((char *)&recvState->image.width, sizeof(recvState->image.width));
    buf->read((char *)&recvState->image.height, sizeof(recvState->image.height));
    buf->read((char *)&recvState->image.compressedSize, sizeof(recvState->image.compressedSize));
    recvState->image.data.resize(recvState->image.compressedSize);
    buf->read((char *)recvState->image.data.data(), recvState->image.compressedSize);
    recvState->image.updated = true;
    unlock();
    sync[SyncPoints::RecvImage].cv.notify_all();
  }
}

void MiniRR::handleWriteMessage(async::message_pointer message)
{
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
      //rr.sendSize(1024,768);
    }
  } else {
    minirr::MiniRR rr;
    rr.initAsServer(31050);
    rr.run();
  }
}
#endif
