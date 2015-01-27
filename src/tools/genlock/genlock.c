/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <linux/config.h>
#include <linux/version.h>
#if defined(CONFIG_MODVERSIONS) && !defined(MODVERSIONS)
#define MODVERSIONS /* force it on */
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 5, 0)
/* only compile as a module */
#define MODULE

/* we are in kernel space */
#define __KERNEL__
#endif

#ifdef MODVERSIONS
//#  include <linux/modversions.h>
#include <config/modversions.h>
#endif

#ifdef CONFIG_SMP
#define __SMP__
#endif

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/delay.h>
#include <linux/serial_reg.h>
#include <linux/ioport.h>
#include <linux/proc_fs.h>
#include <linux/pci.h>
#include <linux/pnp.h>
#include <linux/parport.h>
#include <linux/string.h>
#include <asm/signal.h>
#include <asm/io.h>
#include <asm/uaccess.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 5, 0)
#define pci_for_each_dev(x)                                   \
    for ((x) = pci_find_device(PCI_ANY_ID, PCI_ANY_ID, NULL); \
         (x);                                                 \
         (x) = pci_find_device(PCI_ANY_ID, PCI_ANY_ID, (x)))
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 5, 0)
typedef void irqreturn_t;
#define IRQ_NONE
#define IRQ_HANDLED
#endif

#define DRIVER_NAME "genlock"

#define MAX_PROC_WRITE 1024
#define MAX_PROC_READ 1024

#define log(level, format, arg...)                  \
    do                                              \
    {                                               \
        if (debug >= level)                         \
            printk(DRIVER_NAME ": " format, ##arg); \
    } while (0)

#define sgl_printf(prefix, level, format, ...) (sgl_printf_begin(0), sgl_printf_end(printf("%s[%s]: " format "\n", prefix, sgl_log_msg(level), ##__VA_ARGS__)))

static void small_adjust_do_tasklet(unsigned long dummy);
DECLARE_TASKLET(small_adjust_tasklet, small_adjust_do_tasklet, 0);

static struct proc_dir_entry *proc_genlock = NULL;

static int frametime = -1; /* time of a frame in micro seconds */

static int small_adjust = 0;
static int htotal_orig = -1;
static int htotal_cur = -1;
static int vtotal_orig = -1;
static int vtotal_cur = -1;
static int vnend_not_adjust_counter = 0;
static int adjustment_done = 0;
static int adjust_vnend = 0;
static int vga_small_adjust_length = 250;
static int vga_small_adjust_before_vnend = 220; /* 130, 190 */
static int vga_small_adjust_forbid = 5; /* 3, 10 */
static int vga_small_adjust_pixels = 1; /* 2, 1 */
static int vga_small_adjust_pixelclock = 1;
static int vga_flip_page_before_vnend = 100;
static int vga_hnsize = 1;
static int vga_in_large_adjust = 0;

static int stereoinit = 0;
static int stereosync = 0;
static int stereoflip = 0;
static int current_page = 0;
static int synced = 0;
static int last_drift = 0;
static int last_diff = 0;
static int stereo_equal_counter = 0;
static int my_stereo_state = 0;

static int avg_no_adjust = 50;

static int orig_pixelclock = -1;

static int lp_pnp = 0;

/* statistics */
static unsigned my_frame_counter = 0;
static int master_frame_counter = 0;
static int nv_num_interrupts = 0;
static int lp_num_interrupts = 0;
static int num_small_adjust = 0;
static int num_large_adjust = 0;

#define GENLOCK_DISABLED 0
#define GENLOCK_INIT 1
#define GENLOCK_MEASURE 2
#define GENLOCK_SYNC 3
static int genlock_state = GENLOCK_DISABLED;

static struct timeval tv_lp_irq;
static struct timeval tv_nv_irq;
static struct timeval tv_expected_vnend;

//EXPORT_NO_SYMBOLS;

MODULE_AUTHOR("Martin Aumueller <aumueller@hlrs.de>");
MODULE_DESCRIPTION("hook into NVidia interrupt");
MODULE_LICENSE("GPL");

static int debug = 2;
MODULE_PARM(debug, "i");
MODULE_PARM_DESC(debug, "Debug level");

static int nv_irq = 16;
MODULE_PARM(nv_irq, "i");
MODULE_PARM_DESC(nv_irq, "NVidia interrupt");

static int nv_head = 1;
MODULE_PARM(nv_head, "i");
MODULE_PARM_DESC(nv_head, "Head of NVidia base graphics port");

static int pixelclock = 0;
MODULE_PARM(pixelclock, "i");
MODULE_PARM_DESC(pixelclock, "Use modification of pixelclock for adjustment");

static int lp_irq = -1;
MODULE_PARM(lp_irq, "i");
MODULE_PARM_DESC(lp_irq, "Parallel port interrupt");

static int lp_base = 0x378;
MODULE_PARM(lp_base, "i");
MODULE_PARM_DESC(lp_base, "Parallel port base address");

static int master = 0;
MODULE_PARM(master, "i");
MODULE_PARM_DESC(master, "Master");

static int pageflip = 0;
MODULE_PARM(pageflip, "i");
MODULE_PARM_DESC(pageflip, "Flip page on graphics interrupt");

static int bytesperpixel = 4;
MODULE_PARM(bytesperpixel, "i");
MODULE_PARM_DESC(bytesperpixel, "Number of bytes per pixel (for page flipping)");

/* static int leftoffset = 2800 - 2048; */
static int leftoffset = 0;
MODULE_PARM(leftoffset, "i");
MODULE_PARM_DESC(leftoffset, "Pixel offset for left eye image (for page flipping)");

/* static int rightoffset = 2800 - 1024; */
static int rightoffset = 1024;
MODULE_PARM(rightoffset, "i");
MODULE_PARM_DESC(rightoffset, "Pixel offset for right eye image (for page flipping)");

static int sharedirq = 1;
MODULE_PARM(sharedirq, "i");
MODULE_PARM_DESC(sharedirq, "Interrupt for graphics card is shared");

static int lp_stereo = 0;
MODULE_PARM(lp_stereo, "i");
MODULE_PARM_DESC(lp_stereo, "stereo signal on printer port");

static int ddc_stereo = 0;
MODULE_PARM(ddc_stereo, "i");
MODULE_PARM_DESC(ddc_stereo, "stereo signal on VGA ddc pin");

static int lp_genlock = 1;
MODULE_PARM(lp_genlock, "i");
MODULE_PARM_DESC(lp_genlock, "frame synchronization via printer port");

/* code copied from parport_pc.c */
static const struct pnp_device_id parport_pc_pnp_tbl[] = {
    /* Standard LPT Printer Port */
    {.id = "PNP0400", .driver_data = 0 },
    /* ECP Printer Port */
    {.id = "PNP0401", .driver_data = 0 },
    {}
};

MODULE_DEVICE_TABLE(pnp, parport_pc_pnp_tbl);

static int parport_pc_pnp_probe(struct pnp_dev *dev, const struct pnp_device_id *id)
{
    struct parport *pdata;
    unsigned long io_lo, io_hi;
    int dma, irq;

    if (pnp_port_valid(dev, 0) && !(pnp_port_flags(dev, 0) & IORESOURCE_DISABLED))
    {
        io_lo = pnp_port_start(dev, 0);
    }
    else
        return -EINVAL;

#if 0
        if (pnp_port_valid(dev,1) &&
                !(pnp_port_flags(dev,1) & IORESOURCE_DISABLED)) {
                io_hi = pnp_port_start(dev,1);
        } else
                io_hi = 0;
#endif

#if 0
        if (pnp_irq_valid(dev,0) &&
                !(pnp_irq_flags(dev,0) & IORESOURCE_DISABLED)) {
                irq = pnp_irq(dev,0);
        } else
                irq = PARPORT_IRQ_NONE;
#endif

#if 0
        if (pnp_dma_valid(dev,0) &&
                !(pnp_dma_flags(dev,0) & IORESOURCE_DISABLED)) {
                dma = pnp_dma(dev,0);
        } else
                dma = PARPORT_DMA_NONE;
#endif

#if 0
        printk(KERN_INFO "parport: PnPBIOS parport detected.\n");
        if (!(pdata = parport_pc_probe_port (io_lo, io_hi, irq, dma, NULL)))
                return -ENODEV;

        pnp_set_drvdata(dev,pdata);
#endif
    return 0;
}

static void parport_pc_pnp_remove(struct pnp_dev *dev)
{
    struct parport *pdata = (struct parport *)pnp_get_drvdata(dev);
    if (!pdata)
        return;

    /* parport_pc_unregister_port(pdata); */
}

/* we only need the pnp layer to activate the device, at least for now */
static struct pnp_driver parport_pc_pnp_driver = {
    .name = "genlock_parport_pc",
    .id_table = parport_pc_pnp_tbl,
    .probe = parport_pc_pnp_probe,
    .remove = parport_pc_pnp_remove,
};
/* end of code copied from parport_pc.c */

/* code copied from softgenlock2.0a1 */
static char sgl_nv_device[256] = "01:00.0";

typedef int sgl_iomapping;

#define SGL_NV0_CRTC_ADR 0x00600000
#define SGL_NV1_CRTC_ADR 0x00602000

#define SGL_NV0_VPLL_ADR 0x00680508
#define SGL_NV1_VPLL_ADR 0x00680520

#define SGL_NV0_PCIO_ADR 0x00601000
#define SGL_NV1_PCIO_ADR 0x00603000

unsigned long sgl_nv_ressource = 0;
static unsigned long sgl_nv_res_size = 0x01000000;

static int sgl_nv_nbinit = 0;
//static void* sgl_nv_io = NULL;
static sgl_iomapping sgl_nv_iomap;

static int sgl_nv_crtc_adr = SGL_NV0_CRTC_ADR;
static int sgl_nv_vpll_adr = SGL_NV0_VPLL_ADR;
static int sgl_nv_pcio_adr = SGL_NV0_PCIO_ADR;

#define sgl_mmio_inb(map, ofs) readb((map) + (ofs))
#define sgl_mmio_inw(map, ofs) readw((map) + (ofs))
#define sgl_mmio_inl(map, ofs) readl((map) + (ofs))

#define sgl_mmio_outb(map, val, ofs) writeb((val), (map) + (ofs))
#define sgl_mmio_outw(map, val, ofs) writew((val), (map) + (ofs))
#define sgl_mmio_outl(map, val, ofs) writel((val), (map) + (ofs))

#define sgl_nv_inb(ofs) sgl_mmio_inb(sgl_nv_iomap, ofs)
#define sgl_nv_inw(ofs) sgl_mmio_inw(sgl_nv_iomap, ofs)
#define sgl_nv_inl(ofs) sgl_mmio_inl(sgl_nv_iomap, ofs)
#define sgl_nv_outb(val, ofs) sgl_mmio_outb(sgl_nv_iomap, val, ofs)
#define sgl_nv_outw(val, ofs) sgl_mmio_outw(sgl_nv_iomap, val, ofs)
#define sgl_nv_outl(val, ofs) sgl_mmio_outl(sgl_nv_iomap, val, ofs)

static int sgl_ioremap(const char *devid, int resid, int size, sgl_iomapping *mapping)
{
    int found = 0;
    struct pci_dev *dev;
    pci_for_each_dev(dev)
    {
        if (!strnicmp(devid, dev->dev.bus_id, 1024))
        {
            found = 1;
            break;
        }
    }
    if (!found)
    {
        log(1, "device id \"%s\" not found.", devid);
        log(3, "available devices:");
        pci_for_each_dev(dev)
        {
            log(3, "  id:\"%s\" name:\"%s\"", dev->dev.bus_id, /* dev->name */ "");
        }
        return -2;
    }

    if (resid >= DEVICE_COUNT_RESOURCE || pci_resource_start(dev, resid) == 0) // || !(dev->resource[resid].flags&IORESSOURCE_IO))
    {
        int i;
        log(1, "resource id %xh not found.", resid);
        log(3, "available resources:");
        for (i = 0; i < DEVICE_COUNT_RESOURCE; i++)
        {
            if (dev->resource[i].start != 0)
                log(3, "  id:%xh name:\"%s\" start=%lxh len=%lxh %s", resid,
                    dev->resource[i].name, pci_resource_start(dev, i),
                    pci_resource_len(dev, i), (pci_resource_flags(dev, i) & IORESOURCE_IO) ? "IO" : "");
        }
        return -3;
    }
    /*
  if (!( pci_resource_flags(dev,resid)&IORESOURCE_IO))
  {
    log(1,"resource %s:%xh not IO.",devid,resid);
    return -4;
  }
  */
    if (size == 0)
        size = pci_resource_len(dev, resid);
    if (size > pci_resource_len(dev, resid))
    {
        log(1, "io mapping size %xh bigger that ressource size (%lxh).", size, pci_resource_len(dev, resid));
        return -5;
    }

    *mapping = (int)ioremap(pci_resource_start(dev, resid), size);

    return 0;
}

static int sgl_iounmap(sgl_iomapping mapping)
{
    iounmap((void *)mapping);
    return 0;
}

static int sgl_nv_init0(void)
{
    if (sgl_nv_nbinit == 0)
    {
        sgl_ioremap(sgl_nv_device, sgl_nv_ressource, sgl_nv_res_size, &sgl_nv_iomap);
    }
    ++sgl_nv_nbinit;
    return 0;
}

static int sgl_nv_quit0(void)
{
    --sgl_nv_nbinit;
    if (sgl_nv_nbinit == 0)
    {
        sgl_iounmap(sgl_nv_iomap);
    }
    return 0;
}

static int sgl_nv_quit(void)
{
    return sgl_nv_quit0();
}

static int sgl_nv_init(void)
{
    sgl_nv_init0();
    sgl_nv_outl((sgl_nv_inl(sgl_nv_crtc_adr + 0x804) & -8) | 0x01, sgl_nv_crtc_adr + 0x804); // NON VGA
    return 0;
}

static int sgl_nv_getpixelclock(void)
{
    return sgl_nv_inl(sgl_nv_vpll_adr);
}

static void sgl_nv_setpixelclock(int pixelclock)
{
    sgl_nv_outl(pixelclock, sgl_nv_vpll_adr);
}

static int sgl_nv_calcpixelclock(int base, int ofs)
{
    return base + (ofs << 8);
}

/* end of code copied from softgenlock2.0a1 */

#define VGA_STATUS_PORT 0x3da /* read crtc register */
#define VGA_VRETRACE_BIT 0x08 /* vertical retrace bit */
#define VGA_RETRACE_BIT 0x01 /* retrace bit (vertical or horizontal) */

#define REG_ACCESS_BEGIN \
    {                    \
        unsigned char reg = vga_crtc_reg();
#define REG_ACCESS_END    \
    vga_crtc_setreg(reg); \
    }

static __inline__ unsigned char vga_crtc_reg(void)
{
    return inb(0x3d4);
}

static __inline__ void vga_crtc_setreg(unsigned char reg)
{
    outb(reg, 0x3d4);
}

static __inline__ void vga_crtc_write(unsigned char reg, unsigned char data)
{
    outb(reg, 0x3D4);
    outb(data, 0x3D5);
}

static __inline__ unsigned char vga_crtc_read(unsigned char reg)
{
    outb(reg, 0x3D4);
    return inb(0x3D5);
}

static __inline__ void vga_crtc_write_m(unsigned char reg, unsigned char data, unsigned char mask)
{
    vga_crtc_write(reg, (vga_crtc_read(reg) & ~mask) | (data & mask));
}

static void vga_crtc_unlock(void)
{
    REG_ACCESS_BEGIN
    vga_crtc_write_m(0x11, 0x00, 0x80);
    REG_ACCESS_END
}

static __inline__ unsigned char vga_status(void)
{
    return inb(VGA_STATUS_PORT);
}

/* wait for vertical retrace to start */
static void vga_wait_vr(void)
{
    while (!(vga_status() & VGA_VRETRACE_BIT))
        ;
}

/* wait for vertical retrace to finish */
static void vga_wait_vrend(void)
{
    while (vga_status() & VGA_VRETRACE_BIT)
        ;
}

static void vga_wait_vn(void)
{
    int n;

    do
    {
        n = vga_hnsize;
        while (!(vga_status() & VGA_RETRACE_BIT))
            ;
        while ((vga_status() & VGA_RETRACE_BIT) && n)
            n--;
    } while (n);
}

static int vga_wait_vnend(void)
{
    int n;
    n = vga_hnsize;

    while (n && (vga_status() & VGA_RETRACE_BIT))
        n--;

    if (n <= 0)
    {
        while ((vga_status() & VGA_RETRACE_BIT))
            ;
        return 1;
    }

    return 0;
}

static void vga_init_hnsize(void)
{
    int i;
    int j = 0;

    vga_wait_vr();
    vga_wait_vnend();

    for (i = 0; i < 1000; i++)
    {
        int size = 0;
        // wait for vn
        while (!(vga_status() & VGA_RETRACE_BIT))
            ;
        // wait for hn end & count duration
        while ((vga_status() & VGA_RETRACE_BIT))
            ++size;
        //    sgen_printf("size=%d\n",size);

        // We are looking for x consecutive identical  no retrace sizes :
        //  the horizontal no retrace
        if (j < 20)
        {
            if ((vga_hnsize <= 1 ? 1 : vga_hnsize - 1) <= size && size <= vga_hnsize + 1)
            {
                j++;
            }
            else
            {
                j = 0;
                vga_hnsize = size;
            }
        }
    }

    log(1, "vga_hnsize=%d\n", vga_hnsize);
    //vga_hnsize *= 1.5;
    vga_hnsize += vga_hnsize / 2;
}

static void vga_setstartaddress(int ofs)
{
    static int first = 1;
    // originally I supported higher that 0x40000 offsets
    // but it requires access to NVIDIA specific registers
    // implying some instabilities
    //  int pan=(ofs&3)<<1;
    REG_ACCESS_BEGIN
    ofs = ofs >> 2;
    // this access is for less than 1024 octets offset
    // for 1024xX modes, it never appends
    //  To minimize the number of register access (and so the chance of problem with
    // the driver), I only set the bits 8-15 of the offset.
    if (first || (ofs & 255))
    {
        first = 0;
        vga_crtc_write(0x0D, ofs);
    }
    ofs >>= 8;
    vga_crtc_write(0x0C, ofs);
    //ofs>>=8;
    // we do not write the low 2 bits offset to reduce
    // register access
    //(void)vga_status(); // reset attribute flipflop
    //vga_attr_write(0x13,pan);

    // all this desactivation gives us only 1 access to
    // registers, this is minimal :)
    REG_ACCESS_END
}

static void vga_page_flip(int page)
{
    vga_setstartaddress((page ? rightoffset : leftoffset) * bytesperpixel);
}

static int vga_htotal(void)
{
    int htotal;
    REG_ACCESS_BEGIN
    htotal = vga_crtc_read(0x00); // /8 -5
    REG_ACCESS_END
    return htotal;
}

static void vga_set_htotal(int htotal)
{
    if (htotal_orig == -1)
        htotal_orig = vga_htotal();

    REG_ACCESS_BEGIN
    //vga_reset(1);
    vga_crtc_write(0x00, htotal);
    //vga_reset(0);
    REG_ACCESS_END

    htotal_cur = htotal;
}

/* apparently no resolution larger than 1023 lines in height supported */
static int vga_vtotal(void)
{
    int vtotal;

    REG_ACCESS_BEGIN
    vtotal = vga_crtc_read(0x06);
    vtotal |= ((int)(vga_crtc_read(0x07) & 0x01)) << 8; // bit 8
    REG_ACCESS_END

    return vtotal;
}

/* apparently no resolution larger than 1023 lines in height is supported */
static void vga_set_vtotal(int vtotal)
{
    if (vtotal_orig == -1)
        vtotal_orig = vga_vtotal();

    REG_ACCESS_BEGIN
    vga_crtc_write(0x06, vtotal);
    vga_crtc_write_m(0x07, vtotal >> 8, 0x01); // bit 8
    REG_ACCESS_END

    vtotal_cur = vtotal;
}

static void vga_small_adjust(int us)
{
    if (!pixelclock)
    {
        if (htotal_orig == -1)
            htotal_orig = vga_htotal();

        if (us < 0)
        {
            vga_set_htotal(htotal_orig + vga_small_adjust_pixels);
        }
        else
        {
            vga_set_htotal(htotal_orig - vga_small_adjust_pixels);
        }

        udelay(vga_small_adjust_length);

        vga_set_htotal(htotal_orig);
    }
    else
    {
        int clock = sgl_nv_getpixelclock();
        int newclock = clock + us / 1000;
        if (us < 0)
        {
            newclock = orig_pixelclock + 1;
        }
        else
        {
            newclock = orig_pixelclock - 1;
        }
        sgl_nv_setpixelclock(newclock);

        udelay(vga_small_adjust_length);

        log(1, "vga_small_adjust(%d): oldclock=%d, newclock=%d\n", us, clock, newclock);
        sgl_nv_setpixelclock(orig_pixelclock);
    }
}

static void vga_start_large_adjust(int us)
{
    if (!vga_in_large_adjust)
        log(2, "LARGE ADJUST by %d\n", us);

    if (vtotal_orig == -1)
        vtotal_orig = vga_vtotal();

    vga_in_large_adjust = 1;

    if (us < 0)
    {
        vga_set_vtotal(vtotal_orig + 1);
    }
    else
    {
        vga_set_vtotal(vtotal_orig - 1);
    }
}

static void vga_stop_large_adjust(void)
{
    if (vga_in_large_adjust)
    {
        vga_set_vtotal(vtotal_orig);
        vga_in_large_adjust = 0;
    }
}

static void vga_adjust(int us)
{
    static unsigned too_late_counter = 0;
    static unsigned too_early_counter = 0;
    static unsigned way_off_counter = 0;
    static unsigned no_adjust_counter = 0;

    log(6, "vga_adjust(%d)\n", us);

    if (vga_in_large_adjust && (abs(us) < frametime / 100 || abs(us) < 50))
    {
        vga_stop_large_adjust();
    }

    if (us < 0)
    {
        if (too_early_counter < 30)
            too_early_counter++;
        too_late_counter = 0;
    }
    else
    {
        too_early_counter = 0;
        if (too_late_counter < 30)
            too_late_counter++;
    }

    small_adjust = 0;

    if (abs(us) > 200 && way_off_counter > 10)
    {
        if (too_late_counter > 10 || too_early_counter > 10)
        {
            vga_start_large_adjust(us);
            num_large_adjust++;
            no_adjust_counter++;
            small_adjust = 0;
            too_late_counter = 0;
            too_early_counter = 0;
        }
#if 1
    }
    else if (abs(us) > 80)
    {
        if (abs(us) > 200)
            way_off_counter++;
        else
            way_off_counter = 0;
        if (no_adjust_counter >= vga_small_adjust_forbid && (too_late_counter >= 2 || too_early_counter >= 2))
        {
            small_adjust = us;
        }
    }
    else if (abs(us) > 50)
    {
        if (no_adjust_counter >= avg_no_adjust / 3 && no_adjust_counter >= vga_small_adjust_forbid && (too_late_counter >= 2 || too_early_counter >= 2))
        {
            small_adjust = us;
        }
    }
    else if (abs(us) > 30)
    {
        if (no_adjust_counter >= avg_no_adjust && no_adjust_counter >= vga_small_adjust_forbid && (too_late_counter >= 3 || too_early_counter >= 3))
        {
            small_adjust = us;
        }
#else
    }
    else if (abs(us) > 50)
    {
        small_adjust = us;
#endif
    }

    if (small_adjust)
    {
        log(3, "small adjustment, drift=%d\n", small_adjust);
        tasklet_schedule(&small_adjust_tasklet);
        no_adjust_counter = 0;
        num_small_adjust++;
    }
    else
    {
        no_adjust_counter++;
    }
}

/* set stereo state on DDC pin (for ELSA Revelator cable */
static void ddc_set_stereo_state(int state)
{
    /* NVIDIA specific, but does not seem to work on my Geforce 4 */

    REG_ACCESS_BEGIN

    unsigned char nv_last = vga_crtc_read(0x1f);
    vga_crtc_write(0x1f, 0x57);
    vga_crtc_write_m(0x3f, (state << 4) | 0x01, (1 << 4) | 0x0f);
    if (nv_last != 0x57)
        vga_crtc_write(0x1f, nv_last);

    REG_ACCESS_END
}

#define DATA 0
#define STATUS 1
#define CONTROL 2

static int genlock_lp_init(void)
{
    struct resource *res = request_region(lp_base, 3, DRIVER_NAME);
    if (!res)
    {
        /* region is busy */
        log(0, "parallel io port 0x%x already in use\n", lp_base);
        return -1;
    }

    return 0;
}

static void genlock_lp_free(void)
{
    release_region(lp_base, 3);
}

static void lp_trigger_slave_interrupt(void)
{
    unsigned char data;

    data = inb(lp_base + DATA);

    /* up to 6 slaves can have their ACK pin (pin 10) connected to 
	 * D0-D5 (pins 2-7) in order receive interrupts */
    data |= 0x3f;
    outb(data, lp_base + DATA);
    udelay(80);
    data &= 0xc0;
    outb(data, lp_base + DATA);
}

static void lp_enable_stereo_power(void)
{
    unsigned char data;

    data = inb(lp_base + DATA);

    /* set D2 HIGH */
    data |= 0x04;

    outb(data, lp_base + DATA);
}

static void lp_disable_stereo_power(void)
{
    unsigned char data;

    data = inb(lp_base + DATA);

    /* set D2 LOW */
    data &= ~0x04;

    outb(data, lp_base + DATA);
}

static void lp_set_stereo_state(int state)
{
    /* up to 6 slaves can have their PE pin (pin 12) connected to 
	 * D6,D7 (pins 8,9) and C0-C3 (pins 1,14,16,17)
	 * in order receive the stereo signal */
    unsigned char data, control;

    data = inb(lp_base + DATA);

    if (lp_genlock)
    {
        /* distribute stereo signal to slaves */

        control = inb(lp_base + CONTROL);

        /* control bits C0,C1,C3 are inverted on the output pins */
        if (state)
        {
            /* set the pins HIGH */
            data |= 0xc0;
            control |= 0x04;
            control &= 0xf4;
        }
        else
        {
            /* set the pins LOW */
            data &= 0x3f;
            control &= 0xfb;
            control |= 0x0b;
        }

        outb(control, lp_base + CONTROL);
    }
    else
    {
        /* output stereo sync to glasses on parallel port:
		 * pin 3 (D1): stereo sync
		 * pin 4 (D2): stereo power
		 * pin 18: ground */
        if (state)
        {
            data |= 0x04;
            data |= 0x02;
        }
        else
        {
            data |= 0x04;
            data &= ~0x02;
        }
    }

    outb(data, lp_base + DATA);
}

static int lp_my_stereo_state(void)
{
    unsigned char status = inb(lp_base + STATUS);
    int state = (status & 0x10 ? 1 : 0);
    log(10, "my_stereo_state=%d\n", state);

    /* the local stereo SYNC signal is connected to pin 13
	 * (maps to bit 4 of status register) of the parallel port */
    return (status & 0x10 ? 1 : 0);
}

static int lp_master_stereo_state(void)
{
    unsigned char status = inb(lp_base + STATUS);
    int state = (status & 0x20 ? 1 : 0);
    log(10, "master_stereo_state=%d\n", state);

    /* the master stereo SYNC signal is connected to pin 12
	 * (maps to bit 5 of status register) of the parallel port */
    return (status & 0x20 ? 1 : 0);
}

static __inline__ int tv_diff(const struct timeval *tv_later, const struct timeval *tv_sooner)
{
    return (tv_later->tv_usec - tv_sooner->tv_usec)
           + (tv_later->tv_sec - tv_sooner->tv_sec) * 1000000;
}

static __inline__ void tv_add(struct timeval tv, int diff, struct timeval *res)
{
    res->tv_sec = tv.tv_sec + diff / 1000000;
    res->tv_usec = tv.tv_usec + (diff % 1000000);
}

static void genlock_flip_page(void)
{
    current_page = !current_page;
    vga_page_flip(current_page);

    if (lp_stereo)
        lp_set_stereo_state(current_page);
    if (ddc_stereo)
        ddc_set_stereo_state(current_page);
}

static void small_adjust_do_tasklet(unsigned long dummy)
{
    int diff;
    struct timeval tv_now, tv_vnend;
    int change;

    do_gettimeofday(&tv_now);
    if (pageflip)
    {
        diff = tv_diff(&tv_expected_vnend, &tv_now);
        diff -= vga_flip_page_before_vnend;
        if (diff > 0)
            udelay(diff);
        genlock_flip_page();
    }
    else if (adjust_vnend || vnend_not_adjust_counter > 100)
    {
        log(4, "no adjustment, only adjusting vnend\n");
        vnend_not_adjust_counter = 0;
        adjustment_done = 0;
        adjust_vnend = 0;
    }
    else
    {
        diff = tv_diff(&tv_expected_vnend, &tv_now);
        if (diff > 1500)
        {
            log(1, "would delay too long: %d us\n", diff);
            return;
        }
        if (diff - vga_small_adjust_length - vga_small_adjust_before_vnend >= 0)
        {
            udelay(diff - vga_small_adjust_length - vga_small_adjust_before_vnend);
            vga_small_adjust(small_adjust);
            adjustment_done = 1;
            num_small_adjust++;
            log(5, "correction by %d, delay=%d\n", small_adjust, diff);
        }
        else
        {
            log(1, "no correction: missed: needed time=%d, available time=%d\n",
                vga_small_adjust_length + vga_small_adjust_before_vnend, diff);
            adjustment_done = 0;
        }
    }

    /* resynchronize tv_expected_vnend */
    do_gettimeofday(&tv_now);
    vga_wait_vnend();
    do_gettimeofday(&tv_vnend);
    diff = tv_diff(&tv_vnend, &tv_now); /* gets incremented by frametime in nv_interrupt */
    tv_expected_vnend = tv_vnend;
    if (abs(diff - vga_small_adjust_before_vnend) > 10)
        log(3, "diff between adjust and vnend=%d us\n", diff);

    if (!master && lp_genlock && synced && !stereoflip)
    {
        change = (lp_my_stereo_state() == lp_master_stereo_state()) * 2 - 1;
        if (stereo_equal_counter * change < 100)
            stereo_equal_counter += change;
    }
}

static irqreturn_t nv_interrupt(int int_pt_regs, void *p, struct pt_regs *regs)
{
    static int measure_calls = 0;
    static struct timeval tv_start;
    static struct timeval tv_expected_irq;
    struct timeval tv_now;
    int diff;
    int drift = 0;
    static int too_late_counter = 0, too_early_counter = 0;
    static int irqskipped = 0;
    static int out_of_sync_counter = 0;
    static int lastdiff = 0;
    static int stereo_parity;

    do_gettimeofday(&tv_now);
    nv_num_interrupts++;

    /* guess if this is an interrupt just before a retrace */
    if (sharedirq && genlock_state == GENLOCK_SYNC)
    {
        diff = tv_diff(&tv_now, &tv_expected_irq);
        log(5, "diff between expected and real interrupt: %d us\n", diff);

        if (diff < -1000)
        {
            /* this probably isn't yet the interrupt we are waiting for */
            log(3, "probably not late enough -- still waiting for %d us\n", -diff);
            irqskipped++;
            lastdiff = diff;
            return IRQ_NONE;
        }
        else if (diff > 1000)
        {
            log(1, "PROBABLY OUT OF SYNC - time between expected and real irq: %d us, frametime %d us, %d skipped, lastdiff %d us\n",
                diff, frametime, irqskipped, lastdiff);
            out_of_sync_counter++;
        }

        if (diff < -100)
        {
            too_early_counter++;
            too_late_counter = 0;
        }
        else if (diff > 100)
        {
            too_early_counter = 0;
            too_late_counter++;
        }
        else
        {
            too_early_counter = 0;
            too_late_counter = 0;
        }

        if (out_of_sync_counter > 1000 || diff > frametime)
        {
            //genlock_state = GENLOCK_INIT;
            out_of_sync_counter = 0;
        }

        if (too_early_counter > 10)
        {
            tv_add(tv_expected_irq, diff / 2, &tv_expected_irq);
        }
        else if (too_late_counter > 10)
        {
            tv_add(tv_expected_irq, diff / 2, &tv_expected_irq);
        }
    }

    irqskipped = 0;
    my_frame_counter++;
    tv_nv_irq = tv_now;
    if (num_small_adjust)
    {
        avg_no_adjust = my_frame_counter / num_small_adjust;
    }
    else
    {
        avg_no_adjust = my_frame_counter;
    }

    tv_add(tv_expected_irq, frametime, &tv_expected_irq);
    tv_add(tv_expected_vnend, frametime, &tv_expected_vnend);
    vnend_not_adjust_counter++;

    log(5, "ME: %d, MASTER: %d\n", lp_my_stereo_state(), lp_master_stereo_state());

    switch (genlock_state)
    {
    case GENLOCK_DISABLED:
        return IRQ_NONE;
    case GENLOCK_INIT:
        if (!master && htotal_orig != -1)
        {
            if (abs(vga_htotal() - htotal_orig) <= 1)
                vga_set_htotal(htotal_orig);
        }
        if (!master && vtotal_orig != -1)
        {
            if (abs(vga_vtotal() - vtotal_orig) <= 1)
                vga_set_vtotal(vtotal_orig);
        }
        vga_init_hnsize();
        htotal_orig = vga_htotal();
        vtotal_orig = vga_vtotal();
        frametime = -1;
        measure_calls = 0;
        vga_wait_vr();
        vga_wait_vrend();
        //vga_wait_vnend();
        do_gettimeofday(&tv_start);
        vga_wait_vr();
        vga_wait_vrend();
        do_gettimeofday(&tv_now);
        frametime = tv_diff(&tv_now, &tv_start);
        tv_add(tv_now, frametime, &tv_expected_irq);
        genlock_state = GENLOCK_SYNC;
        return IRQ_NONE;
    case GENLOCK_SYNC:
        if (pageflip && stereosync)
        {
            //genlock_flip_page();
            tasklet_schedule(&small_adjust_tasklet);
        }
        if (master)
        {
            if (!lp_genlock)
                return IRQ_NONE;
            if (stereosync)
            {
                /* my_stereo_state is state during last frame */
                lp_set_stereo_state(!my_stereo_state);
            }
            lp_trigger_slave_interrupt();
            //udelay(200);
            my_stereo_state = lp_my_stereo_state();
            return IRQ_NONE;
        }

        /* calculate time drift from master */
        diff = tv_diff(&tv_nv_irq, &tv_lp_irq);
        last_diff = diff;

        if (stereosync)
        {
            if (stereo_equal_counter < -50)
            {
                stereoflip = 1;
                stereo_equal_counter = 0;
                stereo_parity = (master_frame_counter % 2 != my_frame_counter % 2);
                if (diff > frametime / 2)
                    stereo_parity = !stereo_parity;
            }
        }

        if (stereoflip)
        {
            /* we are adjusting by one whole frame */
            if ((master_frame_counter % 2 == my_frame_counter % 2) == stereo_parity)
            {
                drift = abs(diff) < abs(diff - 2 * frametime)
                            ? diff
                            : diff - 2 * frametime;
            }
            else
            {
                drift = abs(diff + frametime) < abs(diff - frametime)
                            ? diff + frametime
                            : diff - frametime;
            }
            log(2, "doing stereoflip, drift=%d\n", drift);
            if (abs(drift) < frametime / 20)
            {
                log(1, "drift=%d, stereoflip finished\n", drift);
                stereoflip = 0;
                stereo_equal_counter = 0;
            }
        }
        else
        {
            drift = abs(diff) < abs(diff - frametime)
                        ? diff
                        : diff - frametime;
        }
        last_drift = drift;

        if (abs(drift) > 3 * frametime)
        {
            log(3, "drift %d too large -- ignoring\n", drift);
        }
        else
        {
            if (abs(drift))
                synced = 1;
            else
                synced = 0;
            vga_adjust(drift);
        }
        return IRQ_NONE;
    default:
        log(0, "undefined state %d\n", genlock_state);
        return IRQ_NONE;
    }

    return IRQ_NONE;
}

static irqreturn_t lp_interrupt(int int_pt_regs, void *p, struct pt_regs *regs)
{
    static int irreg_interrupt_count = 0;
    struct timeval tv_now;
    int diff;

    lp_num_interrupts++;
    do_gettimeofday(&tv_now);

    diff = tv_diff(&tv_now, &tv_lp_irq);
#if 1
    if (frametime != -1 && abs(frametime - diff) > frametime / 30)
    {
        log(1, "master is probably using different refresh rate - time between parallel irqs: %d us, frametime %d us\n",
            diff, frametime);
        irreg_interrupt_count++;
#if 0
		if(irreg_interrupt_count > 3) {
			log(1, "too many irregular parallel interrupts - disabling\n");
			genlock_state = GENLOCK_DISABLED;
		}
#endif
    }
    else
    {
        irreg_interrupt_count = 0;
    }
#endif
    master_frame_counter++;

    tv_lp_irq = tv_now;

    return IRQ_HANDLED;
}

static void genlock_lp_enable_interrupt(void)
{
    unsigned char control;

    /* enable_irq(lp_irq); */

    control = inb(lp_base + CONTROL);
    control |= 0x10;
    outb(control, lp_base + CONTROL);
}

static void genlock_lp_disable_interrupt(void)
{
    unsigned char control;
    control = inb(lp_base + CONTROL);
    control &= 0xef;
    outb(control, lp_base + CONTROL);

    /* disable_irq(lp_irq); */
}

static int proc_read_ports(char *page, char **start, off_t off,
                           int count, int *eof, void *data)
{
    int len = 0;

    len += sprintf(page, "%04x - %04x : %s\n", lp_base + DATA, lp_base + CONTROL, "lp");

    return len;
}

static int proc_read_interrupts(char *page, char **start, off_t off,
                                int count, int *eof, void *data)
{
    int len = 0;

    len += sprintf(page + len, "%02d : %s\n", nv_irq, "nvidia");
    if (!master)
        len += sprintf(page + len, "%02d : %s\n", lp_irq, "lp");

    return len;
}

static ssize_t proc_read_state(struct file *file, char *buf, size_t len,
                               loff_t *ppos)
{
    ssize_t res;
    char *state;
    char mybuf[100];

    switch (genlock_state)
    {
    case GENLOCK_INIT:
        state = "init";
        break;
    case GENLOCK_SYNC:
        state = "sync";
        break;
    case GENLOCK_MEASURE:
        state = "measure";
        break;
    case GENLOCK_DISABLED:
        state = "disabled";
        break;
    default:
        state = "UNDEFINED";
        break;
    }

    snprintf(mybuf, sizeof(mybuf), "%s: %s [%s]",
             master ? "master" : "slave",
             state,
             stereosync ? "stereo" : "mono");
    if (*ppos <= strlen(mybuf))
    {
        res = snprintf(buf, len, "%s\n", mybuf + *ppos);
        *ppos += res;
    }
    else
    {
        res = 0;
    }

    return res;
}

static ssize_t proc_write_state(struct file *file, const char *buffer,
                                size_t count, loff_t *ppos)
{
    char kbuf[MAX_PROC_WRITE + 1];
    int res;

    if (count > MAX_PROC_WRITE)
        return -EINVAL;
    if (copy_from_user(&kbuf, buffer, count))
        return -EFAULT;

    kbuf[count + 1] = '\0';

    if (!strncmp(kbuf, "disabled", 8) || !strncmp(kbuf, "disable", 7))
    {
        if (htotal_orig != -1)
            vga_set_htotal(htotal_orig);
        if (vtotal_orig != -1)
            vga_set_vtotal(vtotal_orig);
        genlock_state = GENLOCK_DISABLED;
    }
    else if (!strncmp(kbuf, "init", 4))
    {
        genlock_state = GENLOCK_INIT;
    }
    else if (!strncmp(kbuf, "stereo", 6))
    {
        if (genlock_state == GENLOCK_DISABLED)
            genlock_state = GENLOCK_INIT;
        stereoflip = 0;
        stereoinit = 1;
        stereosync = 1;
        stereo_equal_counter = 0;
    }
    else if (!strncmp(kbuf, "mono", 4))
    {
        if (genlock_state == GENLOCK_DISABLED)
            genlock_state = GENLOCK_INIT;
        stereoflip = 0;
        stereosync = 0;
        stereoinit = 0;
        stereo_equal_counter = 0;
    }
    else if (!strncmp(kbuf, "flip", 4) || !strncmp(kbuf, "stereoflip", 10))
    {
        //stereo_parity = !stereo_parity;
        stereoflip = 1;
        stereo_equal_counter = 0;
    }
    else
    {
        log(1, "unknown state \"%s\"\n", kbuf);
    }

    res = count;
    return res;
}

struct file_operations proc_state_operations = {
    read : proc_read_state,
    write : proc_write_state
};

static ssize_t proc_read_config(struct file *file, char *buf, size_t size,
                                loff_t *ppos)
{
    char kbuf[MAX_PROC_READ + 1];
    ssize_t len = 0;
    ssize_t res = 0;

    len += sprintf(kbuf + len, "adjustlength: %d\n", vga_small_adjust_length);
    len += sprintf(kbuf + len, "adjustbeforevnend: %d\n", vga_small_adjust_before_vnend);
    len += sprintf(kbuf + len, "adjustforbid: %d\n", vga_small_adjust_forbid);
    if (pixelclock)
    {
        len += sprintf(kbuf + len, "adjustpixelclock: %d\n", vga_small_adjust_pixelclock);
    }
    else
    {
        len += sprintf(kbuf + len, "adjustpixels: %d\n", vga_small_adjust_pixels);
    }

    if (*ppos <= len)
    {
        strncpy(buf, kbuf + *ppos, size);
        res = (*ppos + size < len) ? size : len - *ppos;
        *ppos += res;
    }
    else
    {
        res = 0;
    }

    return res;
}

static ssize_t proc_write_config(struct file *file, const char *buffer,
                                 size_t count, loff_t *ppos)
{
    char kbuf[MAX_PROC_WRITE + 1];
    int res;
    char *p;
    int val;

    if (count > MAX_PROC_WRITE)
        return -EINVAL;
    if (copy_from_user(&kbuf, buffer, count))
        return -EFAULT;

    kbuf[count + 1] = '\0';

    p = strchr(kbuf, '=');
    if (p == NULL)
    {
        log(1, "invalid config term \"%s\"\n", kbuf);
        return 0;
    }
    *p = '\0';
    p++;
    val = simple_strtol(p, NULL, 0);

    if (!strcmp(kbuf, "adjustlength"))
    {
        vga_small_adjust_length = val;
    }
    else if (!strcmp(kbuf, "adjustbeforevnend"))
    {
        vga_small_adjust_before_vnend = val;
    }
    else if (!strcmp(kbuf, "adjustpixels"))
    {
        vga_small_adjust_pixels = val;
    }
    else if (!strcmp(kbuf, "adustpixelclock"))
    {
        vga_small_adjust_pixelclock = val;
    }
    else if (!strcmp(kbuf, "adjustforbid"))
    {
        vga_small_adjust_forbid = val;
    }
    else if (!strcmp(kbuf, "debug"))
    {
        debug = val;
    }
    else if (!strcmp(kbuf, "pageflip"))
    {
        pageflip = val;
    }
    else if (!strcmp(kbuf, "lp_stereo"))
    {
        lp_stereo = val;
    }
    else
    {
        log(1, "unknown value \"%s\"\n", kbuf);
    }

    res = count;
    return res;
}

struct file_operations proc_config_operations = {
    read : proc_read_config,
    write : proc_write_config
};

static int proc_read_options(char *page, char **start, off_t off,
                             int count, int *eof, void *data)
{
    int len = 0;

    len += sprintf(page + len, "debug: %d\n", debug);
    len += sprintf(page + len, "master: %d\n", master);
    len += sprintf(page + len, "pageflip: %d\n", pageflip);
    len += sprintf(page + len, "sharedirq: %d\n", sharedirq);

    return len;
}

static int proc_read_info(char *page, char **start, off_t off,
                          int count, int *eof, void *data)
{
    int len = 0;

    len += sprintf(page + len, "Frametime: ");
    if (frametime != -1)
        len += sprintf(page + len, "0.%06d s\n", frametime);
    else
        len += sprintf(page + len, "n/a\n");
    if (pixelclock)
    {
        len += sprintf(page + len, "Original pixel clock: ");
        if (orig_pixelclock != -1)
            len += sprintf(page + len, "%d\n", orig_pixelclock);
        else
            len += sprintf(page + len, "n/a\n");
    }

    len += sprintf(page + len, "Graphics interrupts: %d\n", nv_num_interrupts);
    if (sharedirq)
        len += sprintf(page + len, "Graphics interrupts at vertical retrace: %d\n", my_frame_counter);
    if (lp_genlock)
    {
        len += sprintf(page + len, "Original height: ");
        if (vtotal_orig != -1)
            len += sprintf(page + len, "%d\n", vtotal_orig);
        else
            len += sprintf(page + len, "n/a\n");
        len += sprintf(page + len, "Original width: ");
        if (htotal_orig != -1)
            len += sprintf(page + len, "%d\n", htotal_orig);
        else
            len += sprintf(page + len, "n/a\n");
    }
    if (!master)
    {
        len += sprintf(page + len, "Printer interrupts: %d\n",
                       lp_num_interrupts);
        len += sprintf(page + len, "Small adjustments: %d\n",
                       num_small_adjust);
        len += sprintf(page + len, "Large adjustments: %d\n",
                       num_large_adjust);
        len += sprintf(page + len, "Average frames without adjustment: %d\n",
                       avg_no_adjust);
        len += sprintf(page + len, "Last diff: %d.%06d s\n",
                       last_diff / 1000000, last_diff % 1000000);
        len += sprintf(page + len, "Last drift: ");
        if (last_drift >= 0)
        {
            len += sprintf(page + len, "%d.%06d s\n",
                           last_drift / 1000000, last_drift % 1000000);
        }
        else
        {
            len += sprintf(page + len, "-%d.%06d s\n",
                           -last_drift / 1000000, -last_drift % 1000000);
        }
    }
    if (stereosync && !master)
    {
        len += sprintf(page + len, "Stereo equal counter: %d\n", stereo_equal_counter);
    }

    return len;
}

static int init_genlock(void)
{
    int ret;

    log(1, "loading\n");

    ret = pnp_register_driver(&parport_pc_pnp_driver);
    if (ret >= 0)
    {
        log(1, "register parport via PnP\n");
        lp_pnp = 1;
    }

    if (lp_stereo || lp_genlock)
    {
        ret = genlock_lp_init();
        if (ret)
        {
            log(0, "unable to request parallel port io range (base 0x%x).\n", lp_base);
            return -1;
        }
    }

    if (lp_stereo)
        lp_enable_stereo_power();

    vga_crtc_unlock();

    if (0 == nv_head)
    {
        sgl_nv_crtc_adr = SGL_NV0_CRTC_ADR;
        sgl_nv_vpll_adr = SGL_NV0_VPLL_ADR;
        sgl_nv_pcio_adr = SGL_NV0_PCIO_ADR;
    }
    else if (1 == nv_head)
    {
        sgl_nv_crtc_adr = SGL_NV1_CRTC_ADR;
        sgl_nv_vpll_adr = SGL_NV1_VPLL_ADR;
        sgl_nv_pcio_adr = SGL_NV1_PCIO_ADR;
    }
    else
    {
        log(0, "head for nvidia board has to 0 or 1\n");
        return -1;
    }

    if (pixelclock)
    {
        sgl_nv_init();
        orig_pixelclock = sgl_nv_getpixelclock();
        vga_small_adjust_length = 10;
    }

    ret = request_irq(nv_irq, &nv_interrupt,
                      SA_INTERRUPT | SA_SHIRQ,
                      "genlock", &nv_interrupt);
    if (ret)
    {
        log(0, "unable to get Nvidia IRQ %d (errno=%d).\n", nv_irq, ret);
        if (lp_stereo)
            lp_disable_stereo_power();

        if (lp_stereo || lp_genlock)
        {
            genlock_lp_free();
        }
        return ret;
    }
    log(1, "successfully registered nv_interrupt\n");

    if (!master)
    {
        if (lp_irq == -1)
        {
            switch (lp_base)
            {
            case 0x378:
                lp_irq = 7;
                break;
            case 0x278:
                lp_irq = 2;
                break;
            case 0x3bc:
                lp_irq = 5;
                break;
            default:
                log(0, "uncommon port address for printer port - don't know which irq to chose\n");
                return -1;
            }
        }

        ret = request_irq(lp_irq, &lp_interrupt,
                          SA_INTERRUPT | SA_SHIRQ,
                          "genlock", &lp_interrupt);
        if (ret)
        {
            log(0, "unable to get IRQ %d (errno=%d).\n", lp_irq, ret);
            free_irq(nv_irq, &nv_interrupt);

            if (lp_stereo)
                lp_disable_stereo_power();

            if (lp_stereo || lp_genlock)
            {
                genlock_lp_free();
            }

            return ret;
        }
        genlock_lp_enable_interrupt();
        log(1, "successfully registered lp_interrupt\n");
    }

    if (proc_root_driver)
    {
        struct proc_dir_entry *entry;
        proc_genlock = create_proc_entry(DRIVER_NAME,
                                         S_IFDIR | S_IRUGO | S_IXUGO | S_IWUSR,
                                         proc_root_driver);
        create_proc_read_entry("ports", S_IFREG | S_IRUGO,
                               proc_genlock, proc_read_ports, NULL);
        create_proc_read_entry("options", S_IFREG | S_IRUGO,
                               proc_genlock, proc_read_options, NULL);
        create_proc_read_entry("info", S_IFREG | S_IRUGO,
                               proc_genlock, proc_read_info, NULL);
        create_proc_read_entry("interrupts", S_IFREG | S_IRUGO,
                               proc_genlock, proc_read_interrupts, NULL);
        entry = create_proc_entry("state", S_IFREG | S_IRUGO | S_IWUGO,
                                  proc_genlock);
        if (entry)
            entry->proc_fops = &proc_state_operations;
        entry = create_proc_entry("config", S_IFREG | S_IRUGO | S_IWUGO,
                                  proc_genlock);
        if (entry)
            entry->proc_fops = &proc_config_operations;
    }

    /* enable_irq(nv_irq);
        if(!master)
        {
	        enable_irq(lp_irq);
        }
        */

    return 0;
}

static void cleanup_genlock(void)
{
    tasklet_disable(&small_adjust_tasklet);
#if 0
	tasklet_disable(&page_flip_tasklet);
#endif

    if (pageflip)
    {
        vga_setstartaddress(leftoffset * bytesperpixel);
    }

    /* disable_irq(nv_irq); */

    free_irq(nv_irq, &nv_interrupt);

    if (!master)
    {
        if (vtotal_orig != -1)
            vga_set_vtotal(vtotal_orig);
        if (htotal_orig != -1)
            vga_set_htotal(htotal_orig);

        genlock_lp_disable_interrupt();
        free_irq(lp_irq, &lp_interrupt);
    }

    if (proc_genlock)
    {
        remove_proc_entry("ports", proc_genlock);
        remove_proc_entry("options", proc_genlock);
        remove_proc_entry("info", proc_genlock);
        remove_proc_entry("interrupts", proc_genlock);
        remove_proc_entry("state", proc_genlock);
        remove_proc_entry("config", proc_genlock);
        remove_proc_entry(DRIVER_NAME, proc_root_driver);
    }

    if (lp_stereo)
    {
        lp_disable_stereo_power();
    }

    if (lp_stereo || lp_genlock)
        genlock_lp_free();

    if (lp_pnp)
        pnp_unregister_driver(&parport_pc_pnp_driver);

    if (pixelclock)
    {
        sgl_nv_setpixelclock(orig_pixelclock);
        sgl_nv_quit();
    }
    log(1, "removed\n");
}

module_init(init_genlock);
module_exit(cleanup_genlock);
