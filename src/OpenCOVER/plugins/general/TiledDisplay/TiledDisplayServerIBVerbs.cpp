/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef HAVE_IBVERBS

#include <TiledDisplayServerIBVerbs.h>

#include <cover/coVRPluginSupport.h>

#include "TiledDisplayOGLTexQuadCompositor.h"
#include "TiledDisplayOSGTexQuadCompositor.h"

TiledDisplayServerIBVerbs::TiledDisplayServerIBVerbs(int number)
    : TiledDisplayServer(number)
{
    ib = new IBVerbsTransport();
    ctx = NULL;
    dest = NULL;
    remoteDest = NULL;
    this->number = number;
    once = 0;
}

TiledDisplayServerIBVerbs::~TiledDisplayServerIBVerbs()
{
    delete ib;
    delete ctx;
    delete dest;
    delete remoteDest;
}

bool TiledDisplayServerIBVerbs::accept()
{
    int tx_depth = 1;
    int ib_port = 1;
    int port = 18515 + number;
    int sockfd = ib->server_connect(port);

    char buf[16];
    int bytesRead = 0;

    while (bytesRead < 16)
        bytesRead += read(sockfd, buf + bytesRead, 16 - bytesRead);

    if (sscanf(buf, "%d %d", &width, &height) != 2)
    {
        fprintf(stderr, "exchanging width and height failed\n");
        return false;
    }
    else
    {
        dimension.width = width;
        dimension.height = height;
    }

    int size = width * height * 4;

    //fprintf(stderr, "width %d height %d\n", width, height);

    ctx = ib->init_ctx(size, tx_depth, ib_port, NULL);

    if (!ctx)
    {
        fprintf(stderr, "failed to initialize context\n");
        return false;
    }
    else
    {
        dest = ib->init_dest(ctx, ib_port);
        remoteDest = ib->server_exch_dest(sockfd, dest);
        ib->connect_ctx(ctx, ib_port, dest->psn, remoteDest);
        /* An additional handshake is required *after* moving qp to RTR.
         Arbitrarily reuse exch_dest for this purpose. */
        Destination *d = ib->server_exch_dest(sockfd, dest);
        delete d;
        close(sockfd);
    }

    if (ibv_req_notify_cq(ctx->cq, 0))
    {
        fprintf(stderr, "ibv_req_notify failed!\n");
        return false;
    }

    return true;
}

void TiledDisplayServerIBVerbs::run()
{
    isRunning = true;
    accept();

    struct ibv_cq *ev_cq;
    struct ibv_wc wc;
    void *ev_ctx;
    int ne;

    while (keepRunning)
    {
        if (bufferAvailable)
        {
            sendLock.lock();

            struct ibv_recv_wr *bad_wr;

            struct ibv_sge sge;
            sge.addr = (uintptr_t)ctx->buf;
            sge.length = ctx->size;
            sge.lkey = ctx->mr->lkey;

            struct ibv_recv_wr recv;
            recv.next = NULL;
            recv.wr_id = 3;
            recv.sg_list = &sge;
            recv.num_sge = 1;

            if (ibv_post_recv(ctx->qp, &recv, &bad_wr))
                fprintf(stderr, "Couldn't post recv\n");

            do
            {
                ne = ibv_poll_cq(ctx->cq, 1, &wc);
                if (ne > 0)
                {
                    // completion event. RDMA_WRITE on the remote side finished
                    //fprintf(stderr, "RDMA finished %d\n", ((unsigned char *)ctx->buf)[0]);
                    if (ibv_get_cq_event(ctx->ch, &ev_cq, &ev_ctx))
                    {
                        fprintf(stderr, "Failed to get cq event!\n");
                        return;
                    }
                    if (ev_cq != ctx->cq)
                    {
                        fprintf(stderr, "Unkown CQ!\n");
                        return;
                    }

                    ibv_ack_cq_events(ctx->cq, 1);
                    ibv_req_notify_cq(ctx->cq, 0);
                }
            } while (ne == 0);

            pixels = (unsigned char *)ctx->buf;
            dataAvailable = true;
            bufferAvailable = false;

            sendLock.unlock();
        }
        microSleep(10000);
    }
    isRunning = false;
}

#endif
