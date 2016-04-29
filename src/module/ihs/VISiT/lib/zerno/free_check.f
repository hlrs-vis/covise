C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE FREE_CHECK(coord_num,lnods,komp_e,komp_d,
     *                      kern_kn,kern_kn_adr,
     *                      kern_el,kern_el_adr,
     *                      lapp_el,lapp_el_adr,
     *                      nkern_max,nlapp_el,nlapp_kn,
     *                      zeig,help,error_geb,knpar)

      implicit none     

      include 'common.zer'

      integer  coord_num,lnods,komp_e,komp_d,
     *         kern_kn,kern_kn_adr,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,
     *         nkern_max,nlapp_el,nlapp_kn,
     *         zeig,help,error_geb,knpar

      integer i,k,nnn,igeb,ikn,luerr,ielem,inode

      logical fehler

      dimension coord_num(npoin_max),lnods(nelem_max,nkd),
     *          komp_d(npoin+1),komp_e(nl_kompakt),
     *          zeig(npoin),help(npoin),error_geb(ngebiet),
     *          knpar(npoin)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1)
c     *****************************************************************


c     *****************************************************************
c     KONTROLLE OB ALLE KERNKNOTEN DIE NOTWENDIGEN NACHBARN BESITZEN:
 
      do 101 i=1,npoin
         zeig(i)=0
         help(i)=0
 101  continue


      do 100 igeb=1,ngebiet

         error_geb(igeb)=0
         nnn=0

c        Markieren der Knoten der Kernelemente:
         do 110 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            do 120 k=1,nkd
               ikn=lnods(ielem,k)
               if (zeig(ikn).eq.0) then
                  nnn=nnn+1
                  help(nnn)=ikn
                  zeig(ikn)=1
               endif
 120        continue
 110     continue

c        Markieren der Knoten der Ueberlappelemente:
         do 111 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
            do 121 k=1,nkd
               ikn=lnods(ielem,k)
               if (zeig(ikn).eq.0) then
                  nnn=nnn+1
                  help(nnn)=ikn
                  zeig(ikn)=1
               endif
 121        continue
 111     continue


c        Kontrolle der Kernknoten:
         do 210 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            inode=kern_kn(i)
            do 220 k=komp_d(inode),komp_d(inode+1)-1
               ikn=komp_e(k)
               if (zeig(ikn).eq.0) then

c       ccccccccccc
c       write(6,229) igeb,coord_num(inode),coord_num(ikn)
c229    format(2(i7,1x)4x,2(i7,1x))
c       ccccccccccc

                  error_geb(igeb)=error_geb(igeb)+1
               endif
 220        continue
 210     continue


c        Initialisierung des Zeigerfeldes:
         do 102 i=1,nnn
            zeig(help(i))=0
 102     continue

 100  continue
c     *****************************************************************


c     *****************************************************************
c     AUSWERTUNG DES FEHLER-FELDES:

      fehler=.false.
      nnn=0
      do 300 igeb=1,ngebiet
         if (error_geb(igeb).ne.0) then
            fehler=.true.
            nnn=nnn+1
         endif
 300  continue

      if (fehler) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine FREE_CHECK'
         write(luerr,*)'Es gibt Gebiete in denen nicht alle '
         write(luerr,*)'Knoten vorhanden sind, um die Gleichungen '
         write(luerr,*)'der Kernknoten zu formulieren.            '
         write(luerr,*)'Anzahl fehlerhafte Gebiete:',nnn         
         write(luerr,*)'Gebiet   Anzahl fehlender Knoten'
         do 310 igeb=1,ngebiet
            if (error_geb(igeb).ne.0) then
               write(luerr,333) igeb,error_geb(igeb)
            endif
 310     continue
 333     format(1x,i4,8x,i7)
         call erro_ende(myid,parallel,luerr)
      endif
c     *****************************************************************

      return
      end
