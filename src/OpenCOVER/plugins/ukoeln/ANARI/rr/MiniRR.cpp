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

struct RenderState
{
  struct {
    int value{0};
    bool updated{false};
  } numChannels;

  struct {
    PerFrame value;
    bool updated{false};
  } perFrame;

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

void MiniRR::sendNumChannels(const int &numChannels)
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

void MiniRR::sendPerFrame(const PerFrame &perFrame)
{
  lock();
  renderState->perFrame.value = perFrame;
  renderState->perFrame.updated = true;
  unlock();

  auto buf = std::make_shared<Buffer>();
  buf->write((const char *)&renderState->perFrame.value, sizeof(renderState->perFrame.value));
  write(MessageType::SendPerFrame, buf);
}

void MiniRR::recvPerFrame(PerFrame &perFrame)
{
  auto buf = std::make_shared<Buffer>();
  write(MessageType::RecvPerFrame, buf);

  std::unique_lock<std::mutex> l(sync[SyncPoints::RecvPerFrame].mtx);
  sync[SyncPoints::RecvPerFrame].cv.wait(
      l, [this]() { return renderState->perFrame.updated; });
  l.unlock();

  lock();
  perFrame = renderState->perFrame.value;
  renderState->perFrame.updated = false;
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

void MiniRR::sendImage(const uint32_t *img, int width, int height)
{
  lock();

  TurboJPEGOptions options;
  options.width = width;
  options.height = height;
  options.pixelFormat = TurboJPEGOptions::PixelFormat::RGBX;
  options.quality = 80;

  renderState->image.data.resize(getMaxCompressedBufferSizeTurboJPEG(options));

  #ifdef TIMING
  double t0 = getCurrentTime();
  #endif
  size_t compressedSize;
  compressTurboJPEG((const uint8_t *)img,
      renderState->image.data.data(),
      compressedSize,
      options);
  #ifdef TIMING
  double t1 = getCurrentTime();
  std::cout << "compressTurboJPEG takes " << (t1-t0) << " sec.\n";
  #endif

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

  #ifdef TIMING
  double t0 = getCurrentTime();
  #endif
  uncompressTurboJPEG(renderState->image.data.data(),
      (uint8_t *)img,
      (size_t)renderState->image.compressedSize,
      options);
  #ifdef TIMING
  double t1 = getCurrentTime();
  std::cout << "uncompressTurboJPEG takes " << (t1-t0) << " sec.\n";
  #endif

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
  else if (message->type() == MessageType::SendPerFrame) {
    lock();
    buf->read((char *)&renderState->perFrame.value, sizeof(renderState->perFrame.value));
    renderState->perFrame.updated = true;
    sync[SyncPoints::RecvPerFrame].cv.notify_all();
    unlock();
    sync[SyncPoints::RecvPerFrame].cv.notify_all();
  }
  else if (message->type() == MessageType::SendBounds) {
    lock();
    buf->read((char *)renderState->bounds.aabb, sizeof(renderState->bounds.aabb));
    renderState->bounds.updated = true;
    sync[SyncPoints::RecvBounds].cv.notify_all();
    unlock();
    sync[SyncPoints::RecvBounds].cv.notify_all();
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
