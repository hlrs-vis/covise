
#pragma once

// std
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <thread>
// async
#include "async/connection.h"
#include "async/connection_manager.h"
#include "async/work_queue.h"
// ours
#include "Buffer.h"

namespace minirr
{

struct State;

typedef float Mat4[16];
typedef float AABB[6];

struct Viewport
{
  int32_t x{0}, y{0}, width{1}, height{1};
};

struct Camera
{
  Mat4 modelMatrix, viewMatrix, projMatrix;
};

struct Param
{
  std::string name;
  int type;
  uint8_t *value{nullptr};
  unsigned sizeInBytes{0};
};

struct Transfunc
{
  float *rgb{nullptr};
  float *alpha{nullptr};

  uint32_t numRGB{0};
  uint32_t numAlpha{0};

  float absRange[2]{0.f, 1.f};
  float relRange[2]{0.f, 1.f};

  float opacityScale{1.f};
};

struct MiniRR
{
  MiniRR(); 
  ~MiniRR();

  enum Mode { Client, Server, Uninitialized, };

  void initAsClient(std::string hostname, unsigned short port);
  void initAsServer(unsigned short port);

  void run();

  bool connectionClosed();

  void sendNumChannels(const int &numChannels);
  void recvNumChannels(int &numChannels);

  void sendObjectUpdates(const uint64_t &objectUpdates);
  void recvObjectUpdates(uint64_t &objectUpdates);

  void sendViewport(const Viewport &viewport);
  void recvViewport(Viewport &viewport);

  void sendCamera(const Camera &camera);
  void recvCamera(Camera &camera);

  void sendBounds(const AABB &bounds);
  void recvBounds(AABB &bounds);

  void sendAppParams(const Param *params, unsigned numParams);
  void recvAppParams(Param *params, unsigned &numParams);

  void sendTransfunc(const Transfunc &transfunc);
  void recvTransfunc(Transfunc &transfunc);
 
  void sendImage(const uint32_t *img, int width, int height, int jpegQuality);
  void recvImage(uint32_t *img, int &width, int &height, int jpegQuality);

 private:

  std::unique_ptr<State> sendState, recvState;

  Mode mode{Uninitialized};

  async::connection_manager_pointer manager;
  async::connection_pointer conn;
  async::work_queue queue;

  struct MessageType
  {
    enum
    {
      ConnectionEstablished, // internal!
      SendNumChannels,
      RecvNumChannels,
      SendObjectUpdates,
      RecvObjectUpdates,
      SendViewport,
      RecvViewport,
      SendCamera,
      RecvCamera,
      SendBounds,
      RecvBounds,
      SendAppParams,
      RecvAppParams,
      SendTransfunc,
      RecvTransfunc,
      SendImage,
      RecvImage,
      Unknown,
      // Keep last:
      Count,
    };
  };
  typedef MessageType SyncPoints;

  std::mutex globalMtx;
  void lock() { globalMtx.lock(); }
  void unlock() { globalMtx.unlock(); }

  struct SyncPrimitives
  {
    std::mutex mtx;
    std::condition_variable cv;
  };

  SyncPrimitives sync[SyncPoints::Count];

  bool handleNewConnection(async::connection_pointer new_conn, const boost::system::error_code &e);

  void handleMessage(async::connection::reason reason,
      async::message_pointer message,
      const boost::system::error_code &e);

  void handleReadMessage(async::message_pointer message);
  void handleWriteMessage(async::message_pointer message);

  void write(unsigned type, std::shared_ptr<Buffer> buf);

  void write(unsigned type, const void *begin, const void *end);

  void writeImpl(unsigned type, std::shared_ptr<Buffer> buf);
  
  void writeImpl2(unsigned type, const void *begin, const void *end);

};

} // namespace minirr



