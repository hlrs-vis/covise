/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef HAVE_IBVERBS

#include <ParallelRenderingServerIBVerbs.h>
#include "ParallelRenderingCompositor.h"

#include <cover/coVRPluginSupport.h>

ParallelRenderingServerIBVerbs::ParallelRenderingServerIBVerbs(int numClients, bool compositorRenders)
    : ParallelRenderingServer(numClients, compositorRenders)
{

    ib = new IBVerbsTransport();
    ctx = new Context *[numClients];
    dest = new Destination *[numClients];
    remoteDest = new Destination *[numClients];
    dimension = new ParallelRenderingDimension[numClients];

    connected = false;
    front = 1;
    lock.lock();
}

ParallelRenderingServerIBVerbs::~ParallelRenderingServerIBVerbs()
{

    delete ib;

    for (int index = startClient; index < numClients; index++)
    {
        delete ctx[index];
        delete dest[index];
        delete remoteDest[index];
    }

    delete[] ctx;
    delete[] dest;
    delete[] remoteDest;
    delete[] dimension;
}

void ParallelRenderingServerIBVerbs::run()
{

    while (keepRunning)
    {

        lock.lock();
        receive();
    }
}

void ParallelRenderingServerIBVerbs::acceptConnection()
{

    fprintf(stderr, "ParallelRenderingServerIBVerbs::accept()\n");
    for (int index = startClient; index < numClients; index++)
    {

        int tx_depth = 1;
        int ib_port = 1;
        int port = 18515 + index;
        if (!compositorRenders)
            port++;
        int sockfd = ib->server_connect(port);

        char buf[16];
        int bytesRead = 0;

        while (bytesRead < 16)
            bytesRead += read(sockfd, buf + bytesRead, 16 - bytesRead);

        if (sscanf(buf, "%d %d", &width, &height) != 2)
        {
            fprintf(stderr, "exchanging width and height failed\n");
            close(sockfd);
            return;
        }
        else
        {
            dimension[index].width = width;
            dimension[index].height = height;
        }

        int size = width * height * 4;

        ctx[index] = ib->init_ctx(size, tx_depth, ib_port, NULL, NULL);

        if (!ctx[index])
        {
            fprintf(stderr, "failed to initialize context\n");
            return;
        }
        else
        {
            dest[index] = ib->init_dest(ctx[index], ib_port);
            remoteDest[index] = ib->server_exch_dest(sockfd, dest[index]);
            ib->connect_ctx(ctx[index], ib_port, dest[index]->psn, remoteDest[index]);
            /* An additional handshake is required *after* moving qp to RTR.
            Arbitrarily reuse exch_dest for this purpose. */
            Destination *d = ib->server_exch_dest(sockfd, dest[index]);
            delete d;
            close(sockfd);
            fprintf(stderr, "ParallelRenderingServerIBVerbs::accepted()\n");
        }

        if (ibv_req_notify_cq(ctx[index]->cq, 0))
        {
            fprintf(stderr, "ibv_req_notify failed!\n");
            return;
        }
    }

    dimension[0].width = dimension[1].width;
    dimension[0].height = dimension[1].height;
    connected = true;
}

void ParallelRenderingServerIBVerbs::receive()
{
    // receive

    struct ibv_cq **ev_cq = new struct ibv_cq *[numClients];

    struct ibv_wc *wc = new struct ibv_wc[numClients];
    void **ev_ctx = new void *[numClients];

    struct ibv_recv_wr **bad_wr = new struct ibv_recv_wr *[numClients];

    struct ibv_sge *sge = new struct ibv_sge[numClients];
    struct ibv_recv_wr *recv = new struct ibv_recv_wr[numClients];

    for (int index = startClient; index < numClients; index++)
    {

        sge[index].addr = (uintptr_t)ctx[index]->buf1;
        sge[index].lkey = ctx[index]->mr1->lkey;

        sge[index].length = ctx[index]->size;

        recv[index].next = NULL;
        recv[index].wr_id = 3;
        recv[index].sg_list = &sge[index];
        recv[index].num_sge = 1;

        if (ibv_post_recv(ctx[index]->qp, &recv[index], &bad_wr[index]))
            fprintf(stderr, "Couldn't post recv\n");
    }

    char *l = new char[numClients];
    for (int index = startClient; index < numClients; index++)
        l[index] = 0;

    bool finished = false;
    bool error = false;

    while (!finished)
    {

        finished = true;
        for (int index = startClient; index < numClients; index++)
        {

            if (l[index])
                continue;

            int ne = ibv_poll_cq(ctx[index]->cq, 1, &wc[index]);
            if (ne > 0)
            {
                // completion event. RDMA_WRITE on the remote side finished
                // fprintf(stderr, "RDMA on server (%d) finished %d\n", index, ((unsigned char *)ctx[index]->buf)[0]);
                if (ibv_get_cq_event(ctx[index]->ch, &ev_cq[index], &ev_ctx[index]))
                {
                    fprintf(stderr, "Failed to get cq event!\n");
                    error = true;
                    break;
                }
                if (ev_cq[index] != ctx[index]->cq)
                {
                    fprintf(stderr, "Unkown CQ!\n");
                    error = true;
                    break;
                }

                ibv_ack_cq_events(ctx[index]->cq, 1);
                ibv_req_notify_cq(ctx[index]->cq, 0);

                l[index] = 1;
            }
            else
                finished = false;
        }
        microSleep(100);
    }

    delete[] l;
    delete[] ev_cq;
    delete[] wc;
    delete[] ev_ctx;
    delete[] bad_wr;
    delete[] sge;
    delete[] recv;

    renderLock.unlock();
}

void ParallelRenderingServerIBVerbs::render()
{

    renderLock.lock();

    for (int index = startClient; index < numClients; index++)
    {
        if (compositors[index])
        {
            compositors[index]->setTexture(dimension[index].width, dimension[index].height, (unsigned char *)ctx[index]->buf1);
            compositors[index]->render();
        }
    }

    lock.unlock();
}

#endif
