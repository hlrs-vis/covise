C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE DOPP_ELEM(knpar,elpar,coord_num,
     *                     lnods,lnods_num,komp_e,komp_d,
     *                     lapp_kn,lapp_kn_adr,
     *                     lapp_el,lapp_el_adr,
     *                     lapp_el_proz,
     *                     dopp_el,dopp_el_adr,dopp_el_proz,
     *                     nlapp_kn,nlapp_el,ndopp_el,
     *                     zeig_kern,help_dola,zeig_hier,
     *                     error_geb,schreiben)

      implicit none     

      include 'common.zer'

      integer  knpar,elpar,coord_num,
     *         lnods,lnods_num,komp_e,komp_d,
     *         lapp_kn,lapp_kn_adr,
     *         lapp_el,lapp_el_adr,lapp_el_proz,
     *         dopp_el,dopp_el_adr,dopp_el_proz,
     *         nlapp_kn,nlapp_el,ndopp_el,
     *         zeig_kern,help_dola,zeig_hier,
     *         error_geb

      integer i,k,nnn,igeb,ndola,ikn,
     *        luerr,ielem,iproz,ndopp_ges,nlapp_ges,ndopp,
     *        ndata,nfehl,inode,ispal

      logical schreiben,fehler

      dimension knpar(npoin_max),elpar(nelem_max),
     *          coord_num(npoin_max),lnods(nelem_max,nkd),
     *          lnods_num(nelem_max),
     *          komp_d(npoin+1),komp_e(nl_kompakt)

      dimension zeig_kern(npoin_max),help_dola(npoin_max),
     *          zeig_hier(npoin_max),error_geb(ngebiet)

      dimension lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1),
     *          lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_el_proz(nlapp_el)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_el_proz(ndopp_el)

c     *****************************************************************



c     **************************************************************
c     BESTIMMUNG DER DOPP-ELEMENTE:
c     Initialisierungen:
      do 401 i=1,npoin_max
         zeig_hier(i)=0
         zeig_kern(i)=0
         help_dola(i)=0
 401  continue
      do 407 igeb=1,ngebiet
         error_geb(igeb)=0
 407  continue

      dopp_el_adr(1)=1 

      ndopp=0
      do 400 igeb=1,ngebiet

         ndola=0
c     hier werden alle halo-elems markiert (zeig_hier)
c     und alle Knoten der 'eigenen' halo-elems markiert (zeig_kern)
         do 410 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
	    ielem=lapp_el(i)
	    zeig_hier(ielem)=1
	    if(lapp_el_proz(i).eq.igeb)then
	       do 420 k=1,nkd
	          zeig_kern(lnods(ielem,k))=1
 420           continue
            endif
 410     continue
         do 430 i=1,nelem
            nnn=0
	    do 440 k=1,nkd
               ikn=lnods(i,k)
	       if(zeig_kern(ikn).eq.1)then
	         nnn=nnn+1
	       endif
 440        continue
            IF (nnn.ne.0.and.nnn.le.nkd) THEN
c              Element besitzt markierten und anderen Knoten 
               if ((elpar(i).ne.igeb).and.
     *	           (zeig_hier(i).ne.1)) then
c                 Element ist weder Kern- noch halo-Element 
                  ndola=ndola+1
                  help_dola(ndola)=i
                  zeig_hier(i)=1 
	       endif
            ENDIF
 430     continue

         IF (schreiben) THEN
            dopp_el_adr(igeb+1)=dopp_el_adr(igeb)+ndola

            do 450 i=1,ndola
               ndopp=ndopp+1
               dopp_el(ndopp)=help_dola(i)
 450        continue
         ELSE 
            ndopp=ndopp+ndola
         ENDIF


c        Initialisierungen: 
         do 405 i=1,npoin_max
            zeig_hier(i)=0
            zeig_kern(i)=0
 405     continue

 400  continue
c     **************************************************************


c     *****************************************************************
c     BESTIMMUNG DER GEBIETSZUORDNUNG DER DOPPLAPPELEMENTE

      IF (schreiben) THEN

         ndopp_ges=0

         do 300 igeb=1,ngebiet
c           Bestimmung der Gebietsnummer der Dopplappelemente:
            do 320 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
               ielem=dopp_el(i)
               if (elpar(ielem).eq.0) then
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler in Routine DOPP_ELEM'
                  write(luerr,*)'Das Feld elpar ist nicht komplett '
                  write(luerr,*)'Element ',ielem
                  write(luerr,*)'mit Elementnummer ',
     *		                  lnods_num(ielem)
                  write(luerr,*)'hat keine Gebietsnummer.        '
                  call erro_ende(myid,parallel,luerr)
               else
                  iproz=elpar(ielem)
                  dopp_el_proz(i)=iproz
               endif
 320        continue

 300     continue

      ENDIF
c     **************************************************************


c     **************************************************************
c     DIMENSIONSKONTROLLEN:

      IF (schreiben) THEN

        if (ndopp.ne.ndopp_el) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine DOPP_ELEM'
           write(luerr,*)'Die zuvor bestimmte maximale Anzahl an '
           write(luerr,*)'DoppLapp-Elementen ist falsch.              '
           write(luerr,*)'Bestimmte  Anzahl ndopp_el   :',ndopp_el
           write(luerr,*)'Benoetigte Anzahl ndopp      :',ndopp
           call erro_ende(myid,parallel,luerr)
        endif

      ELSE
        ndopp_el=ndopp
      ENDIF
c     **************************************************************

c     **************************************************************

      return
      end
