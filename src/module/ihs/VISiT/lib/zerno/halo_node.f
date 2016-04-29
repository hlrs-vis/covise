C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE HALO_NODE(knpar,lnods,lnods_num,
     *                     coord_num,
     *                     kern_kn,kern_kn_adr,
     *                     lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *                     kern_el,kern_el_adr,
     *                     lapp_el,lapp_el_adr,
     *                     nkern_max,nlapp_el,nlapp_kn,
     *                     zeig_kern,zeig_halo,help_halo,schreiben)

      implicit none     

      include 'common.zer'

      integer  knpar,coord_num,
     *         lnods,lnods_num,
     *         kern_kn,kern_kn_adr,
     *         lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,
     *         nkern_max,nlapp_el,nlapp_kn,
     *         zeig_kern,zeig_halo,help_halo

      integer i,k,igeb,ikn,nlapp,nhalo,luerr,ielem

      logical schreiben

      dimension knpar(npoin_max),coord_num(npoin_max),
     *          lnods(nelem_max,nkd),lnods_num(nelem_max)

      dimension zeig_halo(npoin_max),help_halo(npoin_max),
     *          zeig_kern(npoin_max)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1)

      dimension lapp_kn_proz(nlapp_kn)
c     *****************************************************************


c     **************************************************************
c     BESTIMMUNG DER HALO-KNOTEN:  

c     Initialisierungen:
      do 401 i=1,npoin_max
         zeig_kern(i)=0
         zeig_halo(i)=0
 401  continue


      lapp_kn_adr(1)=1 

      nlapp=0
      do 400 igeb=1,ngebiet

c        Markieren der Kern-Knoten:
         do 410 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            zeig_kern(kern_kn(i))=1
 410     continue


c        Checken, ob alle Knoten der Kernelemente markiert wurden:
         nhalo=0
         do 510 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
            ielem=kern_el(i)
            do 520 k=1,nkd
               ikn=lnods(ielem,k)
               if (zeig_kern(ikn).eq.0) then
                 call erro_init(myid,parallel,luerr)
                 write(luerr,*)'Fehler in Routine HALO_NODE'
                 write(luerr,*)'Die zuvor in KERN_DATA bestimmten '
                 write(luerr,*)'Kernelemente enthalten nicht nur  '
                 write(luerr,*)'Kernknoten!                       '
                 write(luerr,*)'Kernelement :',lnods_num(ielem)   
                 write(luerr,*)'enthaelt Knoten :',coord_num(ikn)
                 call erro_ende(myid,parallel,luerr)
c                  if (zeig_halo(ikn).eq.0) then
cc                    Knoten wurde noch nicht geschrieben:
c                     zeig_halo(ikn)=1
c                     nhalo=nhalo+1
c                     help_halo(nhalo)=ikn
c                  endif
               endif
 520        continue
 510     continue

c        Markieren der Knoten der Halo-Elemente:
         do 530 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
            do 540 k=1,nkd
               ikn=lnods(ielem,k)
               if (zeig_kern(ikn).eq.0) then
                  if (zeig_halo(ikn).eq.0) then
c                    Knoten wurde noch nicht geschrieben:
                     zeig_halo(ikn)=1
                     nhalo=nhalo+1
                     help_halo(nhalo)=ikn
                  endif
               endif
 540        continue
 530     continue

         IF (schreiben) THEN
            lapp_kn_adr(igeb+1)=lapp_kn_adr(igeb)+nhalo

            do 430 i=1,nhalo
               nlapp=nlapp+1
               ikn=help_halo(i)
               lapp_kn(nlapp)=ikn
               lapp_kn_proz(nlapp)=knpar(ikn)
 430        continue
         ELSE 
            nlapp=nlapp+nhalo
         ENDIF


c        Initialisierungen: 
         do 402 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
            zeig_kern(kern_kn(i))=0
 402     continue
         do 403 i=1,nhalo
            zeig_halo(help_halo(i))=0
 403     continue

 400  continue
c     **************************************************************


c     **************************************************************
c     DIMENSIONSKONTROLLEN:

      IF (schreiben) THEN

        if (nlapp.ne.nlapp_kn) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine HALO_NODE'
           write(luerr,*)'Die zuvor bestimmte maximale Anzahl an '
           write(luerr,*)'Halo-Knoten ist falsch.              '
           write(luerr,*)'Bestimmte  Anzahl nlapp_kn   :',nlapp_kn
           write(luerr,*)'Benoetigte Anzahl nlapp      :',nlapp
           call erro_ende(myid,parallel,luerr)
        endif

      ELSE
        nlapp_kn=nlapp
      ENDIF
c     **************************************************************


c     **************************************************************
c     AUSDRUCK:
      
      IF (schreiben) THEN

c        write(lupar,*)'             '         
c        write(lupar,*)'GEBIETS-DIMENSIONEN:'         
c        write(lupar,*)'Gebiet       nkern_kn     nlapp_kn'
c        do 750 igeb=1,ngebiet
c              write(lupar,766) igeb,
c    *             (kern_kn_adr(igeb+1)-kern_kn_adr(igeb)),
c    *             (lapp_kn_adr(igeb+1)-lapp_kn_adr(igeb))
c750     continue
c766     format(i4,10x,i7,6x,i7)
 
c        write(lupar,*)'             '         
c        write(lupar,*)'KNOTEN-DATEN:'         
c        do 701 igeb=1,ngebiet

c           write(lupar,*)'Kern-Knoten von Gebiet',igeb
c           do 711 i=kern_kn_adr(igeb),kern_kn_adr(igeb+1)-1
c              write(lupar,777) kern_kn(i)
c711        continue
 
c           write(lupar,*)'Halo-Elemente von Gebiet',igeb
c           do 721 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
c              write(lupar,777) lapp_kn(i),lapp_kn_proz(i)
c721        continue
c701     continue
 
 777     format(i7,5x,3(i7,1x))

      ENDIF
c     **************************************************************

      return
      end
