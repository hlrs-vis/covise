C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE DOPP_CHECK(lnods_num,lnods,
     *                      elmat,elmat_adr,nl_elmat,
     *                      kern_kn,kern_kn_adr,
     *                      kern_el,kern_el_adr,
     *                      lapp_el,lapp_el_adr,lapp_el_proz,
     *                      dopp_el,dopp_el_adr,dopp_el_proz,
     *                      nkern_max,nlapp_el,ndopp_el,
     *                      elpar,
     *                      zeig,help,error_geb)

      implicit none     

      include 'common.zer'

      integer  lnods_num,lnods,
     *         elmat,elmat_adr,nl_elmat,
     *         kern_kn,kern_kn_adr,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,lapp_el_proz,
     *         dopp_el,dopp_el_adr,dopp_el_proz,
     *         nkern_max,nlapp_el,ndopp_el,
     *         zeig,help,error_geb,elpar

      integer i,k,nnn,igeb,iel,luerr,ielem,inode

      logical fehler

      dimension lnods_num(nelem_max),lnods(nelem_max,nkd),
     *          elmat(nl_elmat),elmat_adr(nelem+1),
     *          zeig(npoin),help(npoin),error_geb(ngebiet),
     *          elpar(nelem_max)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_el_proz(nlapp_el)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_el_proz(ndopp_el)
c     *****************************************************************


c     *****************************************************************
c     KONTROLLE OB ALLE KERNELEMENTE ALLE  NACHBARN BESITZEN:
c     Kernelemente sind hier alle 'eigenen', also elpar=igeb  
 
      do 101 i=1,nelem
         zeig(i)=0
         help(i)=0
 101  continue


      do 100 igeb=1,ngebiet

         error_geb(igeb)=0
         nnn=0

c        Markieren der Kernelemente:
         do 110 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            if (zeig(ielem).eq.0) then
               nnn=nnn+1
               help(nnn)=ielem
               zeig(ielem)=1
            else 
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler1 in DOPPCHECK'
               write(luerr,*)'Element',lnods_num(ielem)
               write(luerr,*)'ist in Gebiet ',igeb
               write(luerr,*)'mindestens doppelt'
               call erro_ende(myid,parallel,luerr)
            endif
 110     continue

c        Markieren der Haloelemente:
         do 111 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
               if (zeig(ielem).eq.0) then
                  nnn=nnn+1
                  help(nnn)=ielem
                  zeig(ielem)=1
               else 
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler2 in DOPPCHECK'
                  write(luerr,*)'Element',lnods_num(ielem)
                  write(luerr,*)'ist in Gebiet ',igeb
                  write(luerr,*)'mindestens doppelt'
                  call erro_ende(myid,parallel,luerr)
               endif
 111     continue

c        Markieren der DoppLappElemente:
         if(parti_geo) then
         do 112 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
            ielem=dopp_el(i)
             if (dopp_el_proz(i).ne.igeb) then
               if (zeig(ielem).eq.0) then
                  nnn=nnn+1
                  help(nnn)=ielem
                  zeig(ielem)=1
               else 
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler3 in DOPPCHECK'
                  write(luerr,*)'Element',lnods_num(ielem)
                  write(luerr,*)'Element',ielem
                  write(luerr,*)'ist in Gebiet ',igeb
                  write(luerr,*)'mindestens doppelt'
                  call erro_ende(myid,parallel,luerr)
               endif
            else 
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler4 in DOPPCHECK'
               write(luerr,*)'Element',lnods_num(ielem)
               write(luerr,*)'ist in Gebiet ',igeb
               write(luerr,*)'Dopplappelement, gehoert'
               write(luerr,*)'aber zu diesem Gebiet!'
               write(luerr,*)'dopp_elproz= ',dopp_el_proz(i)
               write(luerr,*)'elpar(ielem)= ',elpar(ielem)
               call erro_ende(myid,parallel,luerr)
            endif
 112     continue
         endif

c     *****************************************************************


c     *****************************************************************

c        Kontrolle der Kernelemente
         do 210 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            do 220 k=elmat_adr(ielem),elmat_adr(ielem+1)-1
               iel=elmat(k)
               if (zeig(iel).eq.0) then
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler5 in Routine DOPP_CHECK'
                  write(luerr,*)'In Gebiet ',igeb
                  write(luerr,*)'ist Element ',lnods_num(ielem),
     *		                ' Kernelement.'
                  write(luerr,*)'Das Nachbarelement ',
     *                           lnods_num(iel),' fehlt!'
                  call erro_ende(myid,parallel,luerr)
                  error_geb(igeb)=error_geb(igeb)+1
               endif
 220        continue
 210     continue


c        Kontrolle der 'eigenen' Haloelemente
         do 230 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
	    if(lapp_el_proz(i).eq.igeb) then
               do 240 k=elmat_adr(ielem),elmat_adr(ielem+1)-1
                  iel=elmat(k)
                  if (zeig(iel).eq.0) then
                     call erro_init(myid,parallel,luerr)
                     write(luerr,*)'Fehler6 in Routine DOPP_CHECK'
                     write(luerr,*)'In Gebiet ',igeb
                     write(luerr,*)'ist Element ',lnods_num(ielem),
     *		                ' Kernelement.'
                     write(luerr,*)'Das Nachbarelement ',
     *                           lnods_num(iel),' fehlt!'
                     call erro_ende(myid,parallel,luerr)
                     error_geb(igeb)=error_geb(igeb)+1
                  endif
 240           continue
            endif
 230     continue


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
         write(luerr,*)'Fehler in Routine DOPP_CHECK'
         write(luerr,*)'Es gibt Gebiete in denen nicht alle '
         write(luerr,*)'Elemente vorhanden sind, um die Gleichungen '
         write(luerr,*)'der Kernelemente zu formulieren.            '
         write(luerr,*)'Anzahl fehlerhafte Gebiete:',nnn         
         write(luerr,*)'Gebiet   Anzahl fehlender Elemente'
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

