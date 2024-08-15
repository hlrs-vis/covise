
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

struct RenderState;

typedef float Mat4[16];
typedef float AABB[6];

struct MiniRR
{
  MiniRR(); 
  ~MiniRR();

  enum Mode { Client, Server, Uninitialized, };

  void initAsClient(std::string hostname, unsigned short port);
  void initAsServer(unsigned short port);

  void run();

  bool connectionClosed();

  void sendNumChannels(int numChannels);
  void recvNumChannels(int &numChannels);

  void sendBounds(AABB bounds);
  void recvBounds(AABB &bounds);
 
  void sendSize(int w, int h);
  void recvSize(int &w, int &h);

  void sendCamera(Mat4 modelMatrix, Mat4 viewMatrix, Mat4 projMatrix);
  void recvCamera(Mat4 &modelMatrix, Mat4 &viewMatrix, Mat4 &projMatrix);

  void sendImage(const uint32_t *img, int width, int height);
  void recvImage(uint32_t *img, int &width, int &height);

 private:

  std::unique_ptr<RenderState> renderState;

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
      SendBounds,
      RecvBounds,
      SendSize,
      RecvSize,
      SendCamera,
      RecvCamera,
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



