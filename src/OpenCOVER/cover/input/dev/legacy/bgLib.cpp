/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include "serialio.h"
#include "bgLib.h"

RS_ERR rs_err;

#define MAX_TRIES 1000
#define PORT "/dev/ttyd2"

static char Cpy[] = "Copyright (c), BG Systems";

// READ_WRITE

int send_outputs(bglv *bgp)
{
    int st;
    char buf[64];
    long count;

    /*
    *  Pack the numbers into a character string
    */
    st = pack_data(bgp, buf);

    count = (long)strlen(buf);
/*
    *  Simply write the characters to the output serial port (ofd)
    */
#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = WriteFile(bgp->sp_fd, buf, st, &nBytesRead, NULL);
    st = (int)nBytesRead;
#else
    st = write(bgp->sp_fd, buf, strlen(buf));
#endif
    if (st != count)
    {
        printf("error: write()\nOSErr: %d\tcount:  %ld\n", st, count);
    }
    return (st);
}

int check_inputs(bglv *bgp)
{
    int st;
    int i;
    char str[64];

    rs_err.cycles++;
    if (rs_err.cycles % 1000 == 0)
    {
        rs_err.cycles = 0;
        rs_err.thou++;
    }

/*
    *  Read the serial port
    */
#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = ReadFile(bgp->sp_fd, str, bgp->str_len, &nBytesRead, NULL);
    st = (int)nBytesRead;
    if (!bResult)
        st = -1;
#else
    st = read(bgp->sp_fd, str, bgp->str_len);
#endif

    /*
    *  Read error
    */
    if (st < 0)
    {
        rs_err.rd++;
        printf("Major read error\n");
        return (-1);
    }

    i = 0;
    /*
    *  We expect to get str_len characters -- if not, we will re-read
    *  the port 100 times.
    *  If you repeatedly get messages when you are running this code
    *  to the effect that the characters were read after 25 re-tries,
    *  then you are trying to sample too fast.  If you repeatedly get
    *  100 re-tries, then you are probably have no communication at all.
    *  Check the chapter on "trouble shooting".
    */
    while (st != bgp->str_len && i < 100)
    {
        /*
       *  Check for LV824 requesting a handshake -- communication may have been
       *  interupted, so we need to send an 'h' back.
       */
        if (str[0] == 'h')
        {
            printf("Handshake requested (%d)\n", i);
            w_lv(bgp->sp_fd, (char *)"h");
#ifndef __sgi
            sleep(1);
#else
            sginap(10);
#endif
            return (-1);
        }
/**
            sginap(1);
      **/
#ifdef WIN32
        DWORD nBytesRead = 0;
        BOOL bResult;
        bResult = ReadFile(bgp->sp_fd, str, bgp->str_len, &nBytesRead, NULL);
        st = (int)nBytesRead;
        if (!bResult)
            st = -1;
#else
        st = read(bgp->sp_fd, str, bgp->str_len);
#endif
        i++;
    }
#ifdef DEBUG
    if (i > 0)
        printf(" %d read attempts.\n", i);
#endif

    if (str[0] != 'p' || str[bgp->str_len - 1] != '\n')
    {
        printf("Unexpected string :  %s\n", str);
        rs_err.rd++;
        return (-1);
    }
    if (str[0] == '\n')
        return (1);

    /*
    *  Aha.  We got some real data !  So convert it from characters
    *  to meaningful numbers, and put them in the bgp data structure.
    */
    st = convert_serial(bgp, str);

    return st;
}

int r_lv(bglv *bgp)
{
    int st;
    int i = 0;
    char str[64];

    str[0] = '\0';
    rs_err.cycles++;
    if (rs_err.cycles % 1000 == 0)
    {
        rs_err.cycles = 0;
        rs_err.thou++;
    }

#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = ReadFile(bgp->sp_fd, str, bgp->str_len, &nBytesRead, NULL);
    st = (int)nBytesRead;
    if (!bResult)
        st = -1;
#else
    st = read(bgp->sp_fd, str, bgp->str_len);
#endif
    if (st < 0)
    {
        rs_err.rd++;
        perror("r_lv() ");
        return (-1);
    }
    while (st != bgp->str_len && i < MAX_TRIES)
    {
/*
            sginap(1);
      */
#ifdef WIN32
        DWORD nBytesRead = 0;
        BOOL bResult;
        bResult = ReadFile(bgp->sp_fd, str, bgp->str_len, &nBytesRead, NULL);
        st = (int)nBytesRead;
        if (!bResult)
            st = -1;
#else
        st = read(bgp->sp_fd, str, bgp->str_len);
#endif
        i++;
    }

    if (i >= 200)
        printf("%d read attempts. You are sampling too fast \n", i);
    if (i >= MAX_TRIES)
        printf("%d read attempts. Dropped Frame \n", i);

    if (st == 0)
    {
        printf("No chars in input buffer\n");
        rs_err.rd++;
        return (-2);
    }
    else if (str[0] != 'B' || str[bgp->str_len - 1] != '\n')
    {
        printf("%d:  %s\n", st, str);
        rs_err.rd++;
        return (-1);
    }

    st = convert_serial(bgp, str);

    return st;
}

#ifdef WIN32
int w_lv(HANDLE sp_fd, char *mode)
#else
int w_lv(int sp_fd, char *mode)
#endif
{
    int st;

#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = WriteFile(sp_fd, mode, (DWORD)strlen(mode), &nBytesRead, NULL);
    st = (int)nBytesRead;
#else
    st = write(sp_fd, mode, strlen(mode));
#endif
    if (st < 0)
        rs_err.wrt++;
    return (st);
}

#ifdef _WIN32
int get_ack(HANDLE sp_fd)
#else
int get_ack(int sp_fd)
#endif
{
    int st;
    int i = 0;
    const int chars = 100;
    char str[chars];

#ifndef _WIN32 // with this it worked on linux
    do
    {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sp_fd, &read_fds);

        // wait for 30 seconds?
        struct timeval tv;
        tv.tv_sec = 30;
        tv.tv_usec = 0;
        st = select(sp_fd + 1, &read_fds, NULL, NULL, &tv);
    } while (st < 0 && errno == EINTR);
    if (st < 0)
    {
        fprintf(stderr, "get_ack(): select error: %s\n", strerror(errno));
        return (-1);
    }
#endif

#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = ReadFile(sp_fd, str, chars, &nBytesRead, NULL);
    st = (int)nBytesRead;
    if (!bResult)
        st = -1;
#else
    st = read(sp_fd, str, chars);
#endif
    if (st < 0)
    {
        fprintf(stderr, "get_ack():  read error: %s\n", strerror(errno));
        return (-1);
    }
    while (st != 2 && i < 200)
    {
        if (st != 0)
            printf("read: %s\n", str);
#ifndef __sgi
        sleep(1);
#else
        sginap(5);
#endif
#ifdef WIN32
        DWORD nBytesRead = 0;
        BOOL bResult;
        bResult = ReadFile(sp_fd, str, chars, &nBytesRead, NULL);
        st = (int)nBytesRead;
        if (!bResult)
            st = -1;
#else
        st = read(sp_fd, str, chars);
#endif
        i++;
    }
    if (i > 2)
        printf("Took %d trys \n", i);
    if (i >= 200)
        printf("Timeout %d chars in buffer \n", st);

    if (str[0] == 'a')
    {
        printf("Setup OK\n");
        return (0);
    }
    else if (str[0] == 'f')
    {
        printf("Setup failed\n");
        return (-1);
    }
    else
    {
        printf("Unexpected response: %s\n", str);
        return (-2);
    }
}

int r_cs(bglv *bgp, char *str)
{
    int st;

    str[0] = '\0';

#ifdef WIN32
    DWORD nBytesRead = 0;
    BOOL bResult;
    bResult = ReadFile(bgp->sp_fd, str, bgp->str_len, &nBytesRead, NULL);
    st = (int)nBytesRead;
    if (!bResult)
        st = -1;
#else
    st = read(bgp->sp_fd, str, bgp->str_len);
#endif
    if (st < 0)
    {
        perror("r_cs() ");
        return (-1);
    }
    if (str[0] == 'C' || str[bgp->str_len - 1] == '\n')
        return (st);
    else
        return (-1);
}

// READ_WRITE END

// PACK

/*
 *  This routine converts the compressed data string from
 *  characters to 8 floats and a single 16 bit int for
 *  the discretes
 */

int pack_data(bglv *bgp, char *out_buf)
{
    int i, k;
    out_buf[0] = 'p';
    i = 1;

    if (bgp->mp_dig_out > 0)
    {
        for (k = 0; k < bgp->n_dig_out / 8; k++)
        {
            out_buf[i++] = ((bgp->mp_dout[k] >> 4) & 0xf) + 0x21;
            out_buf[i++] = (bgp->mp_dout[k] & 0xf) + 0x21;
        }
    }
    else
    {
        /*
       *  Load the 3 digital values into out_buf
       */
        for (k = 2; k >= 0; k--)
        {
            if (bgp->dig_out >> k & 0x10)
            {
                out_buf[i++] = ((bgp->dout[k] >> 4) & 0xf) + 0x21;
                out_buf[i++] = (bgp->dout[k] & 0xf) + 0x21;
            }
        }
    }
    /*
    *  Load the 3 analog values into out_buf
    */
    for (k = 0; k < 3; k++)
    {
        if ((bgp->analog_out >> k) & 0x1)
        {
            out_buf[i++] = ((bgp->aout[k] >> 6) & 0x3f) + 0x21;
            out_buf[i++] = (bgp->aout[k] & 0x3f) + 0x21;
        }
    }
    /*
    *  And terminate the string
    */
    out_buf[i++] = '\n';
    out_buf[i++] = '\0';

    return (0);
}

// PACK END

// OPEN_LV

int open_lv(bglv *bgp, char *p, int flag)
{
    int st;
    char port[4];
    char pt[32];
    char *ep;

    rs_err.wrt = 0;
    rs_err.rd = 0;
    rs_err.len = 0;
    rs_err.nl = 0;
    rs_err.cycles = 0;
    rs_err.thou = 0;

    /*
    * Initialize port.
    */

    ep = getenv("FBPORT");
    if (ep)
        sprintf(pt, "%s", ep);
    else if (p == NULL)
        sprintf(pt, "%s", PORT);
    else
        sprintf(pt, "%s", p);

    port[0] = pt[strlen(pt) - 1];
    bgp->port = atoi(port);
/**
   CAUTION:  If you use FB_BLOCK, an error on the read() call will cause
             the program to block -- potentially forever
        O_NDELAY => no block
   **/
#ifdef WIN32
    bgp->sp_fd = CreateFile(pt,
                            GENERIC_READ | GENERIC_WRITE,
                            0, // must be opened with exclusive-access
                            NULL, // no security attributes
                            OPEN_EXISTING, // must use OPEN_EXISTING
                            0, // not overlapped I/O
                            NULL // hTemplate must be NULL for comm devices
                            );

    if (bgp->sp_fd == INVALID_HANDLE_VALUE)
    {
        // Handle the error.
        printf("could not open com port %s with error %d.\n", pt, GetLastError());
        return (false);
    }
#else
    if (flag == FB_NOBLOCK)
        bgp->sp_fd = open(pt, O_RDWR | O_NDELAY);
    else if (flag == FB_BLOCK)
        bgp->sp_fd = open(pt, O_RDWR);

    if (bgp->sp_fd < 0)
    {
        perror(pt);
        return (-1);
    }
#endif
    st = set_baud(bgp->sp_fd, BAUD192);

    st = check_rev(bgp);
    if (st < 0)
        return (st);
    else
        return (0);
}

#ifdef WIN32
int set_baud(HANDLE sp_fd, int b)
#else
int set_baud(int sp_fd, int b)
#endif
{
#ifdef WIN32
    // Build on the current configuration, and skip setting the size
    // of the input and output buffers with SetupComm.

    DCB dcb;
    BOOL fSuccess;
    fSuccess = GetCommState(sp_fd, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("GetCommState failed with error %d.\n", GetLastError());
        return (false);
    }

    /* First, get the parameters which can be set in the Preferences */
    switch (b & 0x70)
    {
    case BAUD24:
        dcb.BaudRate = CBR_2400;
        break;
    case BAUD48:
        dcb.BaudRate = CBR_4800;
        break;

    case BAUD96:
        dcb.BaudRate = CBR_9600;
        break;

    case BAUD192:
        dcb.BaudRate = CBR_19200;
        break;

    case BAUD384:
        dcb.BaudRate = CBR_38400;
        break;
    case BAUD576:
        dcb.BaudRate = CBR_57600;
        break;

    default:
        dcb.BaudRate = CBR_19200;
        break;
    }
    // Fill in rest of DCB:  8 data bits, no parity, and 1 stop bit.

    dcb.ByteSize = 8; // data size, xmit, and rcv
    dcb.Parity = NOPARITY; // no parity bit
    dcb.StopBits = ONESTOPBIT; // one stop bit

    fSuccess = SetCommState(sp_fd, &dcb);

    if (!fSuccess)
    {
        // Handle the error.
        printf("SetCommState failed with error %d.\n", GetLastError());
        return (false);
    }
    return true;
#else
    struct termios tios;
    int st;

    st = ioctl(sp_fd, TCGETA, &tios);
    tios.c_iflag = IGNBRK | IXON | IXOFF;
    tios.c_oflag = 0;
    tios.c_lflag = ICANON;

#ifdef _OLD_TERMIOS

    switch (b & 0x70)
    {
    case BAUD12:
        tios.c_cflag = B1200 | CS8 | CREAD | CLOCAL;
        break;
    case BAUD24:
        tios.c_cflag = B2400 | CS8 | CREAD | CLOCAL;
        break;
    case BAUD48:
        tios.c_cflag = B4800 | CS8 | CREAD | CLOCAL;
        break;
    case BAUD96:
        tios.c_cflag = B9600 | CS8 | CREAD | CLOCAL;
        break;
    case BAUD192:
        tios.c_cflag = B19200 | CS8 | CREAD | CLOCAL;
        break;
    case BAUD384:
        tios.c_cflag = B38400 | CS8 | CREAD | CLOCAL;
        break;
    }

#else

    switch (b & 0x70)
    {
    case BAUD24:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B2400;
        break;
    case BAUD48:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B4800;
        break;
    case BAUD96:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B9600;
        break;
    case BAUD192:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B19200;
        break;
    case BAUD384:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B38400;
        break;
    case BAUD576:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B57600;
        break;
    case BAUD1152:
        tios.c_cflag = CS8 | CREAD | CLOCAL;
        tios.c_ospeed = B115200;
        break;
    }
#endif

    st = ioctl(sp_fd, TCSETAF, &tios);
    return (st);
#endif
}

void close_lv(bglv *bgp)
{
    int att;

    bgp->baud = BAUD192;

    init_lv(bgp);

    att = 1000 * rs_err.thou + rs_err.cycles;
#ifdef WIN32
    CloseHandle(bgp->sp_fd);
#else
    close(bgp->sp_fd);
#endif
    printf("\nRead Attempts:  %d\n", att);
    printf("\nErrors Detected\n");
    printf("Read        Write    \n");
    printf("%5d      %5d     \n", rs_err.rd, rs_err.wrt);
}

// OPEN_LV END

// INIT

int init_lv(bglv *bgp)
{
    char c1, c2, c3;
    char str[5];
    int st;
    int i;

    st = check_setup(bgp);
    if (st < 0)
        return (st);

    /*
    *  Compute the number of channels requested, and the
    *  appropriate string length.
    */

    /*
    *  Analog inputs
    */
    bgp->n_analog_in = 0;
    for (i = 0; i < 8; i++)
        if ((bgp->analog_in >> i) & 0x1)
            bgp->n_analog_in++;

    /*
    *  Digital inputs
    */

    if (bgp->mp_dig_in == 0)
    {
        switch (bgp->dig_in)
        {
        case 0x0:
            bgp->n_dig_in = 0;
            break;
        case 0x10:
        case 0x20:
        case 0x40:
            bgp->n_dig_in = 8;
            break;
        case 0x30:
        case 0x50:
        case 0x60:
            bgp->n_dig_in = 16;
            break;
        case 0x70:
            bgp->n_dig_in = 24;
            break;
        }
    }
    else
    {
        switch (bgp->mp_dig_in)
        {
        case MD16:
            bgp->n_dig_in = 16;
            break;
        case MD32:
            bgp->n_dig_in = 32;
            break;
        case MD48:
            bgp->n_dig_in = 48;
            break;
        case MD56:
            bgp->n_dig_in = 56;
            break;
        case MD64:
            bgp->n_dig_in = 64;
            break;
        case MD80:
            bgp->n_dig_in = 80;
            break;
        case MD96:
            bgp->n_dig_in = 96;
            break;
        case MD112:
            bgp->n_dig_in = 112;
            break;
        }
    }

    /*
    *  Digital outputs
    */

    if (bgp->mp_dig_out == 0)
    {
        switch (bgp->dig_out)
        {
        case 0x0:
            bgp->n_dig_out = 0;
            break;
        case 0x10:
        case 0x20:
        case 0x40:
            bgp->n_dig_out = 8;
            break;
        case 0x30:
        case 0x50:
        case 0x60:
            bgp->n_dig_out = 16;
            break;
        case 0x70:
            bgp->n_dig_out = 24;
            break;
        }
    }
    else
    {
        switch (bgp->mp_dig_out)
        {
        case MD16:
            bgp->n_dig_out = 16;
            break;
        case MD32:
            bgp->n_dig_out = 32;
            break;
        case MD48:
            bgp->n_dig_out = 48;
            break;
        case MD56:
            bgp->n_dig_out = 56;
            break;
        case MD64:
            bgp->n_dig_out = 64;
            break;
        case MD80:
            bgp->n_dig_out = 80;
            break;
        case MD96:
            bgp->n_dig_out = 96;
            break;
        case MD112:
            bgp->n_dig_out = 112;
            break;
        }
    }

    /*
    *  Analog outputs
    */
    bgp->n_analog_out = 0;
    if (bgp->analog_out > 0)
    {
        for (i = 0; i < 3; i++)
            if ((bgp->analog_out >> i) & 0x1)
                bgp->n_analog_out++;
    }

    /*
    *  Set the string length for receiving data
    */
    bgp->str_len = 2 + (2 * bgp->n_analog_in) + (bgp->n_dig_in / 4);

    /*
    *  First character has the baud rate and the lower 4 analog ins.
    */
    c1 = bgp->baud;
    c1 |= (bgp->analog_in & 0xf);
    /*
    *  Second character has the digital inputs and the upper 4 analog ins
    */
    c2 = bgp->dig_in;
    c2 |= (bgp->analog_in & 0xf0) >> 4;

    if (bgp->Rev.major == 3)
    {
        str[0] = 's';
        /*
       *  Third character (for rev 3 eproms only, has the digital outs (-F)
       *  and analog outs (-3G)
       */
        c3 = bgp->analog_out & 0xf;
        c3 |= bgp->dig_out & 0xf0;

        /*
       *  If multiplex digital inputs selected, overwrite c2 and set c3
       */
        if (bgp->mp_dig_in != 0 || bgp->mp_dig_out != 0)
        {
            c2 = 0;
            if (bgp->mp_dig_in == MD112)
                c2 = MP112i;
            else if (bgp->mp_dig_out == MD112)
                c2 = MP112o;
            else
                c2 = MP56io;

            c2 |= (bgp->analog_in & 0xf0) >> 4;
            c3 |= MPDIG;
        }
        /*
       *  Add the OFFSET to each character to make sure they are not control
       *  characters
       */
        str[1] = c1 + BG_OFFSET;
        str[2] = c2 + BG_OFFSET;
        str[3] = c3 + BG_OFFSET;
        str[4] = '\0';
        st = w_lv(bgp->sp_fd, str);
        /*
       *  Make sure that the LV got the setup !
       */
        st = get_ack(bgp->sp_fd);
        /*
       *  If we have a rev 3.00 eprom, just don't check the return
       *  value - just proceed and assume things are OK.
       *  (Bug fixed in 3.01)
       */
        if (bgp->Rev.bug != 0)
        {
            if (st < 0)
                return (st);
        }
    }
    else if (bgp->Rev.major == 2)
    {
        if (bgp->Rev.minor == 2)
        {
            /*
          *  For rev 2.2 EPROMS use an 'R' and no offset -- so make sure c1 and
          *  c2 are not flow control characters !
          */
            str[0] = 'R';
            str[1] = c1;
            str[2] = c2;
        }
        else if (bgp->Rev.minor >= 3)
        {
            /*
          *  For rev 2.3 EPROMS use an 'r' and offset the characters
          */
            str[0] = 'r';
            str[1] = c1 + BG_OFFSET;
            str[2] = c2 + BG_OFFSET;
        }
        str[3] = '\0';
        st = w_lv(bgp->sp_fd, str);
    }

    st = set_baud(bgp->sp_fd, bgp->baud);

    return (0);
}

// INIT END

// CONVERT

int convert_serial(bglv *bgp, char *str)
{
    int i, digp, j;
    int k = 0;
    float tmp[8];

    digp = 0;

    k = 1 + bgp->n_dig_in / 4;
    if (bgp->mp_dig_in == 0)
    {
        /*
       *  Load the digital input values into dioval
       */
        if (k > 1)
        {
            i = 1;
            for (j = 2; j >= 0; j--)
            {
                if (bgp->dig_in & 0x10 << j)
                {
                    digp = 0x0f & (str[i++] - 0x21);
                    digp = (digp << 4) | (0x0f & (str[i++] - 0x21));
                    bgp->din[j] = digp;
                }
            }
        }
    }
    else
    {
        /*
       *  Load the multiplex digital input array
       */
        i = 1;
        for (j = 0; j < bgp->mp_dig_in / 8; j++)
        {
            digp = 0x0f & (str[i++] - 0x21);
            digp = (digp << 4) | (0x0f & (str[i++] - 0x21));
            bgp->mp_din[j] = digp;
        }
    }
    /*
    *  Load the 8 analog values into inbuf
    */
    for (i = k; i < bgp->str_len - 2; i += 2)
    {
        digp = ((0x3f & (str[i] - 0x21)) << 6) | (0x3f & (str[i + 1] - 0x21));
        tmp[(i - k) / 2] = -1.0f + (2.0f * digp / 4095);
    }
    for (i = 0, k = 0; k < 8; k++)
    {
        if (bgp->analog_in >> k & 0x1)
        {
            bgp->ain[k] = tmp[i];
            i++;
        }
    }

    digp = ((0x0f & (str[22] - 0x21)) << 4) | (0x0f & (str[23] - 0x21));
    return (0);
}

// CONVERT END

// CHECK

int check_rev(bglv *bgp)
{
    int chars_read = 0;
    char str[64];

    /*
    *  Send a "T" and see if the Box responds
    *  Set str_len to 44 for the copyright string.
    */
    w_lv(bgp->sp_fd, (char *)"T");
#ifndef __sgi
    sleep(1);
#else
    sginap(100);
#endif
    bgp->str_len = 44;
    chars_read = r_cs(bgp, str);

    /*
    *  If chars_read <= 0, looks like we have a Rev 1.x EPROM
    */
    if (chars_read <= 0)
    {
        no_answer();
        return (-1);
    }
    else
    {
        /*
       *  Check the string length
       */
        if (chars_read != 44)
        {
            printf("Unexpected characters:  %d  %s\n", chars_read, str);
            return (-1);
        }
        else
        {
            /*
          *  Check that it is the Copyright string
          */
            if (strncmp(str, Cpy, strlen(Cpy)) != 0)
            {
                printf("No match on copyright string:  %d  %s\n", chars_read, str);
                return (-1);
            }
            else
            {
                /*
             *  If we go this far, we should have the right string
             */
                bgp->Rev.year = parse_year(str);
                bgp->Rev.major = str[38] - 48;
                bgp->Rev.minor = str[40] - 48;
                bgp->Rev.bug = str[41] - 48;
                bgp->Rev.alpha = str[42];
                printf("%s %d  Revision %d.%d%d%c\n", Cpy, bgp->Rev.year,
                       bgp->Rev.major, bgp->Rev.minor,
                       bgp->Rev.bug, bgp->Rev.alpha);
            }
        }
    }

    return (bgp->Rev.major);
}

int parse_year(char *s)
{
    int i = 0;
    char yr[12];

    while (*s != '1')
        s++;
    yr[i] = *s;
    while (*s != ' ' && *s != ',')
        yr[i++] = *s++;
    yr[i] = '\0';
    return (atoi(yr));
}

int check_setup(bglv *bgp)
{
    int i;
    int st = 0;

    /*
    *  This routine checks the EPROM revision against the
    *  requested setup, and attempts to identify inconsistencies !
    */

    if (bgp->Rev.major == 2)
    {
        if (bgp->analog_out != 0x0)
        {
            printf("  Analog outputs not supported by LV816\n");
            st = -1;
        }
        if (bgp->dig_out != 0x0)
        {
            printf("  Digital outputs not supported by LV816\n");
            st = -2;
        }
        if (bgp->dig_in & 0x40)
        {
            printf("  Digital inputs 19-24 not supported by LV816\n");
            st = -3;
        }
    }
    else if (bgp->Rev.major == 3)
    {
        switch (bgp->Rev.alpha)
        {
        case 'e':
            printf("LV824-E\n");
            if (bgp->analog_out != 0x0)
            {
                printf("  Analog outputs not supported\n");
                st = -1;
            }
            if (bgp->dig_out != 0x0)
            {
                printf("  Digital outputs not supported\n");
                st = -2;
            }
            break;
        case 'f':
            printf("LV824-F\n");
            if (bgp->analog_out != 0x0)
            {
                printf("  Analog outputs not supported\n");
                st = -2;
            }
            break;
        case 'g':
            printf("LV824-G\n");
            break;
        default:
            st = -3;
            printf("Not an LV824 board\n");
            break;
        }
        if (st < 0)
            return (st);
        /*
       *  Check also for conflict in the digital channels
       */

        if (bgp->dig_in && bgp->dig_out)
        {
            for (i = 0; i < 3; i++)
            {
                if (((bgp->dig_in >> i) & 0x1)
                    && ((bgp->dig_out >> i) & 0x1))
                {

                    printf("Invalid set-up requested.\n");
                    printf("  Digital input group %d AND output group %d selected\n", i + 1, i + 1);

                    printf("\n\n  Digital channels can be set in groups of 8 as\n");
                    printf("  either inputs or outputs.\n");
                    printf("  Of course you can (for example) set the bottom 8\n");
                    printf("  to inputs DIC1 and the top 16 to outputs DOC2 | DOC3\n");

                    st = -5;
                    return (st);
                }
            }
        }
    }
    return (st);
}

void no_answer()
{
    printf("\nWriting a 'T' to the Box produced no answer.  \n");
    printf("\n");
    printf("The expected string was not returned from the BG box.\n");
    printf("Here are some possible problems:\n");
    printf("   1. Check power to Box\n");
    printf("   2. Check the serial cable\n");
    printf("   3. Check the environment variable FBPORT\n");
    printf("      - does it match the connected serial port ?\n");
    printf("   4. Is the serial port configured as a terminal ? \n");
    printf("      - if so use \"System Manager\" to disconnect the port\n");
    printf("   5. You have an old FlyBox (serial no. less than 60) \n");
    printf("         which has a revision 1.0 EPROM.  Call BG Systems.\n");

    printf("\n\n");
}

// CHECK END
