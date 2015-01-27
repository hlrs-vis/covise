/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <unistd.h>
#include <sys/time.h>
#include <netdb.h>
#include <getopt.h>
#include <arpa/inet.h>
#endif
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <util/byteswap.h>
#include <time.h>
#include <errno.h>

#include "IBVerbsTransport.h"

#ifdef HAVE_IBVERBS

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

IBVerbsTransport::IBVerbsTransport()
{

    struct ibv_device **dev_list = ibv_get_device_list(NULL);

    page_size = sysconf(_SC_PAGESIZE);

    ib_dev = dev_list[0];
    if (!ib_dev)
    {
        fprintf(stderr, "No IB devices found\n");
        return;
    }
    else
    {
        fprintf(stderr, "IB device: %s\n", ibv_get_device_name(ib_dev));
    }

    srand48(getpid() * time(NULL));
}

IBVerbsTransport::~IBVerbsTransport()
{
}

Destination *IBVerbsTransport::init_dest(Context *ctx, int ib_port)
{

    Destination *my_dest = new Destination();

    my_dest->lid = get_local_lid(ctx, ib_port);
    my_dest->qpn = ctx->qp->qp_num;
    my_dest->psn = lrand48() & 0xffffff;
    my_dest->rkey = ctx->mr->rkey;
    my_dest->vaddr = (uintptr_t)ctx->buf;

    return my_dest;
}

Context *IBVerbsTransport::init_ctx(unsigned size, int tx_depth,
                                    int port, unsigned char *buf)
{

    Context *ctx = new Context();

    if (!ctx)
        return NULL;

    ctx->size = size;
    ctx->tx_depth = tx_depth;

    if (!buf)
        ctx->buf = memalign(page_size, size);
    else
        ctx->buf = buf;

    if (!ctx->buf)
    {
        fprintf(stderr, "Couldn't allocate work buf.\n");
        return NULL;
    }

    memset(ctx->buf, 0, size);

    ctx->context = ibv_open_device(ib_dev);
    if (!ctx->context)
    {
        fprintf(stderr, "Couldn't get context for %s\n",
                ibv_get_device_name(ib_dev));
        return NULL;
    }

    ctx->pd = ibv_alloc_pd(ctx->context);
    if (!ctx->pd)
    {
        fprintf(stderr, "Couldn't allocate PD\n");
        return NULL;
    }

    /* We dont really want IBV_ACCESS_LOCAL_WRITE, but IB spec says:
    * The Consumer is not allowed to assign Remote Write or Remote Atomic to
    * a Memory Region that has not been assigned Local Write. */
    ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, size,
                         (ibv_access_flags)(IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE));
    if (!ctx->mr)
    {
        fprintf(stderr, "Couldn't allocate MR\n");
        return NULL;
    }

    ctx->ch = ibv_create_comp_channel(ctx->context);
    if (!ctx->ch)
    {
        fprintf(stderr, "Couldn't create comp channel\n");
        return NULL;
    }
    ctx->cq = ibv_create_cq(ctx->context, tx_depth, ctx, ctx->ch, 0);
    if (!ctx->cq)
    {
        fprintf(stderr, "Couldn't create CQ\n");
        return NULL;
    }

    {
        struct ibv_qp_cap cap = {
            tx_depth,
            tx_depth,
            1,
            1,
            0
        };

        struct ibv_qp_init_attr attr = {
            0,
            ctx->cq,
            ctx->cq,
            0,
            cap,
            IBV_QPT_RC,
            0
        };
        /*
      struct ibv_qp_init_attr attr = {
         .send_cq = ctx->cq,
         .recv_cq = ctx->cq,
         .cap     = {
            .max_send_wr  = tx_depth,
            // Work around:  driver doesnt support
            // recv_wr = 0
            .max_recv_wr  = tx_depth,
            .max_send_sge = 1,
            .max_recv_sge = 1,
            .max_inline_data = 0
         },
         .qp_type = IBV_QPT_RC
      };
*/
        ctx->qp = ibv_create_qp(ctx->pd, &attr);
        if (!ctx->qp)
        {
            fprintf(stderr, "Couldn't create QP\n");
            return NULL;
        }
    }

    {
        struct ibv_qp_attr attr;

        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

        if (ibv_modify_qp(ctx->qp, &attr,
                          (ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)))
        {
            fprintf(stderr, "Failed to modify QP to INIT\n");
            return NULL;
        }
    }

    return ctx;
}

int IBVerbsTransport::connect_ctx(Context *ctx,
                                  int port, int my_psn,
                                  Destination *dest)
{

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof attr);

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_2048;
    attr.dest_qp_num = dest->qpn;
    attr.rq_psn = dest->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = dest->lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port;
    if (ibv_modify_qp(ctx->qp, &attr,
                      (ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)))
    {
        fprintf(stderr, "Failed to modify QP to RTR\n");
        return 1;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = my_psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(ctx->qp, &attr,
                      (ibv_qp_attr_mask)(IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)))
    {
        fprintf(stderr, "Failed to modify QP to RTS\n");
        return 1;
    }

    return 0;
}

uint16_t IBVerbsTransport::get_local_lid(Context *ctx, int port)
{

    if (!ctx)
        return 0;

    struct ibv_port_attr attr;

    if (ibv_query_port(ctx->context, port, &attr))
        return 0;

    return attr.lid;
}

int IBVerbsTransport::client_connect(const char *servername, int port)
{

    struct addrinfo *res, *t;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char *service;
    int n;
    int sockfd = -1;

    asprintf(&service, "%d", port);
    n = getaddrinfo(servername, service, &hints, &res);

    if (n < 0)
    {
        fprintf(stderr, "%s for %s:%d\n", gai_strerror(n), servername, port);
        return n;
    }

    fprintf(stderr, "connecting to %s:%d\n", servername, port);

    for (t = res; t; t = t->ai_next)
    {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0)
        {
            if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(sockfd);
            sockfd = -1;
        }
    }

    freeaddrinfo(res);

    if (sockfd < 0)
    {
        fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
        return sockfd;
    }
    else
    {
        fprintf(stderr, "connected to %s:%d\n", servername, port);
    }
    return sockfd;
}

Destination *IBVerbsTransport::client_exch_dest(int sockfd,
                                                Destination *my_dest)
{

    char msg[sizeof "0000:000000:000000:00000000:0000000000000000"];
    int parsed;

    sprintf(msg, "%04x:%06x:%06x:%08x:%016Lx", my_dest->lid, my_dest->qpn,
            my_dest->psn, my_dest->rkey, my_dest->vaddr);
    if (write(sockfd, msg, sizeof msg) != sizeof msg)
    {
        perror("client write");
        fprintf(stderr, "Couldn't send local address\n");
        return 0;
    }

    if (read(sockfd, msg, sizeof msg) != sizeof msg)
    {
        perror("client read");
        fprintf(stderr, "Couldn't read remote address\n");
        return 0;
    }

    Destination *rem_dest = new Destination();

    parsed = sscanf(msg, "%x:%x:%x:%x:%Lx", &rem_dest->lid, &rem_dest->qpn,
                    &rem_dest->psn, &rem_dest->rkey, &rem_dest->vaddr);

    if (parsed != 5)
    {
        fprintf(stderr, "Couldn't parse line <%.*s>\n", (int)sizeof msg,
                msg);
        free(rem_dest);
        return 0;
    }

    return rem_dest;
}

int IBVerbsTransport::server_connect(int port)
{

    struct addrinfo *res, *t;
    struct addrinfo hints;

    memset(&hints, 0, sizeof(hints));

    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char *service;
    int sockfd = -1;
    int n, connfd;

    asprintf(&service, "%d", port);
    n = getaddrinfo(NULL, service, &hints, &res);
    fprintf(stderr, "server_connect %d\n", port);
    if (n < 0)
    {
        fprintf(stderr, "%s for port %d\n", gai_strerror(n), port);
        return n;
    }

    for (t = res; t; t = t->ai_next)
    {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0)
        {
            n = 1;

            setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);

            if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
                break;

            close(sockfd);
            sockfd = -1;
        }
    }

    freeaddrinfo(res);

    if (sockfd < 0)
    {
        fprintf(stderr, "Couldn't listen to port %d\n", port);
        return sockfd;
    }

    fprintf(stderr, "server accept\n");
    listen(sockfd, 1);
    connfd = accept(sockfd, NULL, 0);
    if (connfd < 0)
    {
        perror("error: server accept");
        fprintf(stderr, "accept() failed\n");
        close(sockfd);
        return connfd;
    }

    fprintf(stderr, "server accepted\n");

    close(sockfd);
    return connfd;
}

Destination *IBVerbsTransport::server_exch_dest(int connfd, Destination *my_dest)
{

    char msg[sizeof "0000:000000:000000:00000000:0000000000000000"];
    int parsed;
    int n;

    n = read(connfd, msg, sizeof msg);
    if (n != sizeof msg)
    {
        perror("server read");
        fprintf(stderr, "%d/%d: Couldn't read remote address\n", n, (int)sizeof msg);
        return 0;
    }

    Destination *rem_dest = new Destination();

    parsed = sscanf(msg, "%x:%x:%x:%x:%Lx", &rem_dest->lid, &rem_dest->qpn,
                    &rem_dest->psn, &rem_dest->rkey, &rem_dest->vaddr);
    if (parsed != 5)
    {
        fprintf(stderr, "Couldn't parse line <%.*s>\n", (int)sizeof msg,
                msg);
        free(rem_dest);
        return 0;
    }

    sprintf(msg, "%04x:%06x:%06x:%08x:%016Lx", my_dest->lid, my_dest->qpn,
            my_dest->psn, my_dest->rkey, my_dest->vaddr);
    if (write(connfd, msg, sizeof msg) != sizeof msg)
    {
        perror("server write");
        fprintf(stderr, "Couldn't send local address\n");
        free(rem_dest);
        return 0;
    }

    return rem_dest;
}

#endif
