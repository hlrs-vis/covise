C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE DOPP_NODE(knpar,lnods,lnods_num,
     *                     coord_num,
     *                     lapp_kn,lapp_kn_adr,
     *                     dopp_kn,dopp_kn_adr,dopp_kn_proz,
     *                     lapp_el,lapp_el_adr,
     *                     dopp_el,dopp_el_adr,
     *                     nlapp_kn,nlapp_el,ndopp_el,ndopp_kn,
     *                     zeig_lapp,zeig_halo,help_halo,schreiben)

      implicit none     

      include 'common.zer'

      integer  knpar,coord_num,
     *         lnods,lnods_num,
     *         lapp_kn,lapp_kn_adr,
     *         dopp_kn,dopp_kn_adr,dopp_kn_proz,
     *         lapp_el,lapp_el_adr,
     *         dopp_el,dopp_el_adr,
     *         nlapp_kn,nlapp_el,ndopp_el,ndopp_kn,
     *         zeig_lapp,zeig_halo,help_halo

      integer i,k,igeb,ikn,ndopp,nhalo,luerr,ielem

      logical schreiben

      dimension knpar(npoin_max),coord_num(npoin_max),
     *          lnods(nelem_max,nkd),lnods_num(nelem_max)

      dimension zeig_halo(npoin_max),help_halo(npoin_max),
     *          zeig_lapp(npoin_max)

      dimension lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1),
     *          lapp_el(nlapp_el),lapp_el_adr(ngebiet+1)

      dimension dopp_el(ndopp_el),dopp_el_adr(ngebiet+1),
     *          dopp_kn(ndopp_kn),dopp_kn_adr(ngebiet+1)

      dimension dopp_kn_proz(ndopp_kn)
c     *****************************************************************


c     **************************************************************
c     BESTIMMUNG DER DOPPLAPP-KNOTEN

c     Initialisierungen:
      do 401 i=1,npoin_max
         zeig_lapp(i)=0
         zeig_halo(i)=0
 401  continue


      dopp_kn_adr(1)=1 

      ndopp=0
      do 400 igeb=1,ngebiet

c        Markieren der Lapp-Knoten:
         do 410 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            zeig_lapp(lapp_kn(i))=1
 410     continue


c        Checken, ob alle Knoten der Lappelemente markiert wurden:
         nhalo=0
         do 510 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            ielem=lapp_el(i)
            do 520 k=1,nkd
               ikn=lnods(ielem,k)
               IF ((zeig_lapp(ikn).eq.0).AND.
     *	           (knpar(ikn).ne.igeb)) THEN
                 call erro_init(myid,parallel,luerr)
                 write(luerr,*)'Fehler in Routine DOPP_NODE'
                 write(luerr,*)'Die zuvor in LAPP_NODE bestimmten '
                 write(luerr,*)'HALOelemente enthalten nicht nur  '
                 write(luerr,*)'Kern- und Haloknoten!             '
                 write(luerr,*)'Kernelement :',lnods_num(ielem)   
                 write(luerr,*)'enthaelt Knoten :',coord_num(ikn)
                 call erro_ende(myid,parallel,luerr)
               endif
 520        continue
 510     continue

c        Markieren der Knoten der DOPPLAPP-Elemente
         do 530 i=dopp_el_adr(igeb),dopp_el_adr(igeb+1)-1
            ielem=dopp_el(i)
            do 540 k=1,nkd
               ikn=lnods(ielem,k)
               if (zeig_lapp(ikn).eq.0) then
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
            dopp_kn_adr(igeb+1)=dopp_kn_adr(igeb)+nhalo

            do 430 i=1,nhalo
               ndopp=ndopp+1
               ikn=help_halo(i)
               dopp_kn(ndopp)=ikn
               dopp_kn_proz(ndopp)=knpar(ikn)
 430        continue
         ELSE 
            ndopp=ndopp+nhalo
         ENDIF


c        Initialisierungen: 
         do 402 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            zeig_lapp(lapp_kn(i))=0
 402     continue
         do 403 i=1,nhalo
            zeig_halo(help_halo(i))=0
 403     continue

 400  continue
c     **************************************************************


c     **************************************************************
c     DIMENSIONSKONTROLLEN:

      IF (schreiben) THEN

        if (ndopp.ne.ndopp_kn) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine DOPP_NODE'
           write(luerr,*)'Die zuvor bestimmte maximale Anzahl an '
           write(luerr,*)'DoppLapp-Knoten ist falsch.              '
           write(luerr,*)'Bestimmte  Anzahl ndopp_kn   :',ndopp_kn
           write(luerr,*)'Benoetigte Anzahl ndopp      :',ndopp
           call erro_ende(myid,parallel,luerr)
        endif

      ELSE
        ndopp_kn=ndopp
      ENDIF
c     **************************************************************


c     **************************************************************

      return
      end
