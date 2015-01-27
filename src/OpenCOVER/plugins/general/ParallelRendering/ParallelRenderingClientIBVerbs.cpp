/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef HAVE_IBVERBS

#include <ParallelRenderingClientIBVerbs.h>

ParallelRenderingClientIBVerbs::ParallelRenderingClientIBVerbs(int number, const std::string &compositor)
    : ParallelRenderingClient(number, compositor)
{

    ib = new IBVerbsTransport();
    ctx = NULL;
    dest = NULL;
    remoteDest = NULL;
    this->number = number;
    once = 0;

    lock.lock();
}

ParallelRenderingClientIBVerbs::~ParallelRenderingClientIBVerbs()
{

    delete ib;
    delete ctx;
    delete dest;
    delete remoteDest;
}

void ParallelRenderingClientIBVerbs::run()
{

    struct ibv_wc wc;
    struct ibv_cq *ev_cq;
    void *ev_ctx;

    while (keepRunning)
    {

        lock.lock();
        int ne;

        do
        {
            ne = ibv_poll_cq(ctx->cq, 1, &wc);
            if (ne > 0)
            {
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
            microSleep(100);
        } while (ne == 0);

        if (ne < 0)
        {
            fprintf(stderr, "poll CQ failed %d\n", ne);
            return;
        }
        if (wc.status != IBV_WC_SUCCESS)
        {
            fprintf(stderr, "Completion with error at client\n");
            fprintf(stderr, "Failed status %d: wr_id %d\n",
                    wc.status, (int)wc.wr_id);
            return;
        }
    }
}

void ParallelRenderingClientIBVerbs::connectToServer()
{

    sleep(2);

    int tx_depth = 1;
    int ib_port = 1;
    int port = 18515 + number;

    int sockfd = ib->client_connect(const_cast<char *>(compositor.c_str()), port);

    char buf[16];
    int bytesWritten = 0;
    snprintf(buf, 16, "%d %d", width, height);
    while (bytesWritten < 16)
    {
        int num = write(sockfd, buf + bytesWritten, 16 - bytesWritten);
        if (num > 0)
            bytesWritten += num;
        else
        {
            fprintf(stderr, "ParallelRenderingClientIBVerbs::connect: socket error\n");
            return;
        }
    }

    int size = width * height * 4;

    ctx = ib->init_ctx(size, tx_depth, ib_port, NULL, image);
    fprintf(stderr, "image: [%p]\n", image);

    if (!ctx)
    {
        fprintf(stderr, "failed to initialize context\n");
        return;
    }
    else
    {
        dest = ib->init_dest(ctx, ib_port);

        remoteDest = ib->client_exch_dest(sockfd, dest);
        ib->connect_ctx(ctx, ib_port, dest->psn, remoteDest);
        /* An additional handshake is required *after* moving qp to RTR
         Arbitrarily reuse exch_dest for this purpose. */
        Destination *d = ib->client_exch_dest(sockfd, dest);
        delete d;
        close(sockfd);
    }

    if (ibv_req_notify_cq(ctx->cq, 0))
    {
        fprintf(stderr, "ibv_req_notify failed!\n");
        return;
    }
    connected = true;
}

void ParallelRenderingClientIBVerbs::send()
{

    struct ibv_qp *qp;
    struct ibv_send_wr *bad_wr;

    ctx->list.addr = (uintptr_t)ctx->buf1;
    ctx->list.lkey = ctx->mr1->lkey;
    ctx->wr.wr.rdma.remote_addr = remoteDest->vaddr1;
    ctx->wr.wr.rdma.rkey = remoteDest->rkey1;

    ctx->list.length = ctx->size;

    ctx->wr.wr_id = 3; // id
    ctx->wr.sg_list = &ctx->list;
    ctx->wr.num_sge = 1;
    ctx->wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    ctx->wr.send_flags = IBV_SEND_SIGNALED; // | IBV_SEND_SOLICITED;
    ctx->wr.next = NULL;

    qp = ctx->qp;

    if (ibv_req_notify_cq(ctx->cq, 0))
    {
        fprintf(stderr, "ibv_req_notify failed!\n");
        return;
    }

    if (ibv_post_send(qp, &ctx->wr, &bad_wr))
    {
        fprintf(stderr, "Couldn't post send\n");
        return;
    }

    lock.unlock();
}

#endif
