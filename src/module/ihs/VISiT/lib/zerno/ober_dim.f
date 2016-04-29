C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_DIM(elmat,elmat_adr,elmat_stu,nl_elmat,
     *                    elpar,nelfla,
     *                    kern_ele,kern_adr)
c
      implicit none

      include 'common.zer'
c
      integer elmat,elmat_adr,elmat_stu,nl_elmat,elpar,
     *        kern_ele,kern_adr,
     *        nelfla,nfla_frei,nfla_proz,
     *        i,igeb,iii,nnn,ielem,inach

      dimension elmat(nl_elmat),elmat_stu(nelem+1),elmat_adr(nelem+1),
     *          elpar(nelem_max),kern_ele(nelem),kern_adr(ngebiet+1)
c     ****************************************************************



c     *****************************************************************
c     BERECHNUNG DER ANZAHL FREIER FLAECHEN DER GESAMTGEOMETRIE:

      nfla_frei=0     

c     goto 499
      do 400 i=1,nelem

c        Anzahl stumpf angrenzender Elemente:
         nnn=elmat_stu(i)-elmat_adr(i)+1
         if (nnn.lt.nkd_fla) then
c          Element i besitzt freie Flaechen:
           nfla_frei=nfla_frei+nkd_fla-nnn
         endif
 400  continue
c499  continue
c     *****************************************************************

c     *****************************************************************
c     BERECHNEN DER ANZAHL TRENN-FLAECHEN ZWISCHEN DEN GEBIETEN:

      nfla_proz=0

c     goto 599
      do 500 igeb=1,ngebiet

         do 200 iii=kern_adr(igeb),kern_adr(igeb+1)-1

            ielem=kern_ele(iii)

c           Schleife ueber die Nachbarelemente von Kernelement ielem:
            do 210 i=elmat_adr(ielem),elmat_stu(ielem)
                 inach=elmat(i)

                 IF (elpar(inach).ne.igeb) THEN
c                   Das Element inach grenzt an das Kernelement an
                    nfla_proz=nfla_proz+1
                 ENDIF
 210        continue
 200     continue                     

 500  continue
c599  continue
c     *****************************************************************

      nelfla=nfla_frei+nfla_proz

      return
      end

