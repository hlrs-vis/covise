C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE HALO_ELEM(knpar,elpar,coord_num,
     *                     lnods,lnods_num,komp_e,komp_d,
     *                     kern_kn,kern_kn_adr,
     *                     kern_el,kern_el_adr,
     *                     lapp_el,lapp_el_adr,lapp_el_proz,
     *                     nkern_max,nlapp_el,
     *                     zeig_halo,help_halo,
     *                     error_geb,schreiben)

      implicit none     

      include 'common.zer'

      integer  knpar,elpar,coord_num,
     *         lnods,lnods_num,komp_e,komp_d,
     *         kern_kn,kern_kn_adr,
     *         kern_el,kern_el_adr,
     *         lapp_el,lapp_el_adr,lapp_el_proz,
     *         nkern_max,nlapp_el,
     *         zeig_halo,help_halo,
     *         error_geb

      integer i,k,nnn,igeb,nhalo,ikn,
     *        luerr,ielem,iproz,nlapp_ges,nkern_ges,nlapp,
     *        ndata,nfehl,inode,ispal

      logical schreiben,fehler

      dimension knpar(npoin_max),elpar(nelem_max),
     *          coord_num(npoin_max),lnods(nelem_max,nkd),
     *          lnods_num(nelem_max),
     *          komp_d(npoin+1),komp_e(nl_kompakt)

      dimension zeig_halo(npoin_max),help_halo(npoin_max),
     *          error_geb(ngebiet)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1)

      dimension lapp_el_proz(nlapp_el)
c     *****************************************************************


c     write(lupar,*)'Element 7570'
c     do 111 k=1,nkd
c        write(lupar,*) coord_num(lnods(7570,k)),
c    *                  knpar(lnods(7570,k))
c111  continue
c     do 112 i=1,npoin
c       if (coord_num(i).eq.10786) then
c          write(lupar,*)'i=',i,' coord_num(i)=',coord_num(i),
c    *                ' knpar=',knpar(i)
c       endif
c112  continue


c     **************************************************************
c     BESTIMMUNG DER HALO-ELEMENTE:

c     Initialisierungen:
      do 401 i=1,npoin_max
         zeig_halo(i)=0
 401  continue
      do 407 igeb=1,ngebiet
         error_geb(igeb)=0
 407  continue

      lapp_el_adr(1)=1 

      nlapp=0
      do 400 igeb=1,ngebiet

         nhalo=0
         do 410 i=1,nelem
            nnn=0
            do 420 k=1,nkd
               ikn=lnods(i,k)
               if (knpar(ikn).eq.igeb) then
                  nnn=nnn+1
               endif
 420        continue 

            IF (nnn.ne.0.and.nnn.lt.nkd) THEN
c              Element besitzt Kern- und Halo-Knoten -> Halo-Element
               nhalo=nhalo+1
               help_halo(nhalo)=i
               zeig_halo(i)=1 
            ENDIF
 410     continue


         IF (schreiben) THEN
            lapp_el_adr(igeb+1)=lapp_el_adr(igeb)+nhalo

            do 430 i=1,nhalo
               nlapp=nlapp+1
               lapp_el(nlapp)=help_halo(i)
 430        continue
         ELSE 
            nlapp=nlapp+nhalo
         ENDIF


c        Initialisierungen: 
         do 405 i=1,nhalo
            zeig_halo(help_halo(i))=0
 405     continue

 400  continue
c     **************************************************************


c     *****************************************************************
c     AUSWERTUNG DES FEHLER-FELDES:

      fehler=.false.
      nnn=0
      do 490 igeb=1,ngebiet
         if (error_geb(igeb).ne.0) then
            fehler=.true.
            nnn=nnn+1
         endif
 490  continue

      if (fehler) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine HALO_ELEM'
         write(luerr,*)'Es gibt Gebiete in denen nicht alle '
         write(luerr,*)'Elemente vorhanden sind, um die Gleichungen '
         write(luerr,*)'der Kernknoten zu erfuellen.              '
         write(luerr,*)'Anzahl fehlerhafte Gebiete:',nnn         
         write(luerr,*)'Gebiet   Anzahl fehlender Knoten'
         do 491 igeb=1,ngebiet
            if (error_geb(igeb).ne.0) then
               write(luerr,444) igeb,error_geb(igeb)
            endif
 491     continue
 444     format(1x,i4,8x,i7)
         call erro_ende(myid,parallel,luerr)
      endif
c     *****************************************************************



c     **************************************************************
c     BESTIMMUNG DES ELEMENTPARTITIONIERUNGSFELDES:

      IF (schreiben) THEN
         do 301 i=1,nelem
            elpar(i)=0
 301     continue

         nlapp_ges=0
         do 300 igeb=1,ngebiet

c           Markieren der Kernelemente:
            do 310 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
               ielem=kern_el(i)
               elpar(ielem)=igeb
 310        continue
 300     continue

         do 370 igeb=1,ngebiet
c           Bestimmung der Gebietsnummer der Ueberlappelemente:
            do 320 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
               ielem=lapp_el(i)
               if (elpar(ielem).eq.0) then
                  elpar(ielem)=igeb
                  lapp_el_proz(i)=igeb
                  nlapp_ges=nlapp_ges+1
               else
                  iproz=elpar(ielem)
                  lapp_el_proz(i)=iproz
               endif
 320        continue

 370     continue

c        KONTROLLE OB ALLE ELEMENTE BELEGT WURDEN:
         nnn=0
         do 330 i=1,nelem
            if (elpar(i).eq.0) then
               nnn=nnn+1
            endif
 330     continue

         if (nnn.ne.0) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine HALO_ELEM'
            write(luerr,*)'Das Feld elpar wurde nicht vollstaendig '
            write(luerr,*)'belegt. Dies bedeutet, dass nicht jedem '
            write(luerr,*)'Element eine Gebietsnummer zugewiesen   '
            write(luerr,*)'wurde.                                  '
            write(luerr,*)'Anzahl fehlender Eintraege:',nnn         
            call erro_ende(myid,parallel,luerr)
         endif

      ENDIF
c     **************************************************************


c     **************************************************************
c     DIMENSIONSKONTROLLEN:

      IF (schreiben) THEN

        if (nlapp.ne.nlapp_el) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine HALO_ELEM'
           write(luerr,*)'Die zuvor bestimmte maximale Anzahl an '
           write(luerr,*)'Halo-Elementen ist falsch.              '
           write(luerr,*)'Bestimmte  Anzahl nlapp_el   :',nlapp_el
           write(luerr,*)'Benoetigte Anzahl nlapp      :',nlapp
           call erro_ende(myid,parallel,luerr)
        endif

        nkern_ges=kern_el_adr(ngebiet+1)-kern_el_adr(1)
        if (nkern_ges+nlapp_ges.ne.nelem) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine HALO_ELEM'
           write(luerr,*)'Die Summe aus Kernelementen und Gesamtanzahl'
           write(luerr,*)'Ueberlappelementen muss gleich der          '
           write(luerr,*)'der eingelesenen Elementanzahl sein.    '
           write(luerr,*)'Gesamtanzahl Kernelemente     :',nkern_ges 
           write(luerr,*)'Gesamtanzahl Ueberlappelemente:',nlapp_ges 
           write(luerr,*)'Eingelesene  Elementanzahl    :',nelem        
           call erro_ende(myid,parallel,luerr)
        endif

      ELSE
        nlapp_el=nlapp
      ENDIF
c     **************************************************************


c     **************************************************************
c     AUSDRUCK:
      
c     IF (schreiben) THEN
c
c        write(lupar,*)'             '         
c        write(lupar,*)'GEBIETS-DIMENSIONEN:'         
c        write(lupar,*)'Gebiet       nkern_kn     nlapp_kn   nlapp_noma'
c        do 750 igeb=1,ngebiet
c              write(lupar,766) igeb,
c    *             (kern_el_adr(igeb+1)-kern_el_adr(igeb)),
c    *             (lapp_el_adr(igeb+1)-lapp_el_adr(igeb))
c750     continue
c766     format(i4,10x,i7,6x,i7,6x,i7)
c
c
c        write(lupar,*)'             '         
c        write(lupar,*)'ELEMENT-DATEN:'         
c        do 701 igeb=1,ngebiet
c        do 701 igeb=1,1
c
c           write(lupar,*)'Kern-Elemente von Gebiet',igeb
c           do 711 i=kern_el_adr(igeb),kern_el_adr(igeb+1)-1
c              write(lupar,777) lnods_num(kern_el(i))
c711        continue
c
c           write(lupar,*)'Halo-Elemente von Gebiet',igeb
c           do 721 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
c              write(lupar,777) lnods_num(lapp_el(i)),lapp_el_proz(i)
c721        continue
c701     continue
c
c777     format(i7,5x,3(i7,1x))
c
c     ENDIF
c     **************************************************************

      return
      end
