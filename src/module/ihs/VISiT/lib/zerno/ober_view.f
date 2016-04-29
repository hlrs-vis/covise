C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_VIEW(elpar,lnods,lnods_num,coord,coord_num,
     *                     elmat,elmat_adr,elmat_stu,nl_elmat,
     *                     gemat,gemat_adr,nl_gemat,
     *                     farb_geb,farb_adr,farb_per,nfarb_geb,
     *                     kern_ele,kern_adr,
     *                     ober_name_geo,ober_name_ses,
     *                     isp,lmax,speich_max_sub)
  
      implicit none     

      include 'common.zer'

      integer iende,ifine,speich_max

      integer  elpar,lnods,coord_num,lnods_num,
     *         elmat_adr,elmat_stu,elmat,nl_elmat,
     *         gemat,gemat_adr,nl_gemat,
     *         farb_geb,farb_adr,farb_per,nfarb_geb,
     *         kern_ele,kern_adr,
     *         isp,lmax,speich_max_sub

      integer  nelfla,nelfla_max,nkd_obe,
     *         ielfla_kno,ielfla_ele,ielfla_adr,
     *         ihelp_kno,ihelp_ele,ihelp_num,ipermut

      integer igeb1_help,igeb2_help,igeb3_help,igeb4_help,
     *        iint1_help,iint2_help,igeb

      integer nkern,nnn,i,luerr

      real    coord

      character*80 ober_name_geo,ober_name_ses

      dimension lnods(nelem_max,nkd),elpar(nelem_max),
     *          elmat(nl_elmat),elmat_adr(nelem+1),elmat_stu(nelem+1),
     *          farb_geb(ngebiet),farb_adr(ngebiet+1),
     *          farb_per(ngebiet),
     *          coord_num(npoin_max),lnods_num(nelem_max),
     *          kern_ele(nelem_max),kern_adr(ngebiet+1)

      dimension gemat(nl_gemat),gemat_adr(ngebiet+1),
     *          coord(npoin_max,ncd)

      dimension isp(lmax)
c     *****************************************************************


      iende=1
      ifine=lmax
      speich_max=0

c     *****************************************************************
c     BESTIMMUNG DER KERNELEMENTE AUS DER PARTITIONIERUNG:

      nkern=0
      kern_adr(1)=1
      do 500 igeb=1,ngebiet

         nnn=0
         do 501 i=1,nelem
            if (elpar(i).eq.igeb) then
               nkern=nkern+1
               nnn=nnn+1        
               kern_ele(nkern)=i
            endif
 501     continue

         kern_adr(igeb+1)=kern_adr(igeb)+nnn
 500  continue                 

      if (nkern.ne.nelem) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine OBER_VIEW'
         write(luerr,*)'Die Gesamtanzahl an Kernelementen stimmt mit '
         write(luerr,*)'der Elementanzahl nicht ueberein.            '
         write(luerr,*)'nkern              :',nkern          
         write(luerr,*)'nelem              :',nelem        
         call erro_ende(myid,parallel,luerr)
      endif
c     *****************************************************************

c     *****************************************************************
c     BESTIMMUNG DER FREIEN OBERFLAECHEN UND PROZESSOROBERFLAECHEN:

      CALL ALLOC_FINE(ifine,igeb1_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb2_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb3_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb4_help,ngebiet+1)

      CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
      CALL ALLOC_FINE(ifine,iint2_help,npoin_max)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      if (ncd.eq.3) then
         CALL OBER_DIM(elmat,elmat_adr,elmat_stu,nl_elmat,
     *                 elpar,nelfla,
     *                 kern_ele,kern_adr)
         nkd_obe=nrbknie
      else if (ncd.eq.2) then
         nelfla=nelem
         nkd_obe=nkd        
      endif

      nelfla_max=nelfla

      CALL ALLOC_ENDE(iende,ielfla_adr,ngebiet+1)
      CALL ALLOC_ENDE(iende,ielfla_kno,nelfla_max*nkd_obe)
      CALL ALLOC_ENDE(iende,ielfla_ele,nelfla_max)

      CALL ALLOC_FINE(ifine,ihelp_kno,nelfla_max*nkd_obe)
      CALL ALLOC_FINE(ifine,ihelp_num,nelfla_max)
      CALL ALLOC_FINE(ifine,ihelp_ele,nelfla_max)
      CALL ALLOC_FINE(ifine,ipermut,nelfla_max)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      if (ncd.eq.3) then
         CALL OBER_ADR(lnods,elmat,elmat_adr,elmat_stu,nl_elmat,
     *                 isp(ielfla_kno),isp(ielfla_ele),nelfla,
     *                 nelfla_max,nkd_obe,
     *                 elpar,isp(iint1_help),
     *                 kern_ele,kern_adr,coord,coord_num)
      else if (ncd.eq.2) then
         CALL OBER_2DE(lnods,isp(ielfla_kno),isp(ielfla_ele),
     *                 nelfla,nelfla_max,nkd_obe,
     *                 kern_ele,kern_adr,coord_num)
      endif


c     Sortieren der Elemente nach der Gebietsnummer:
      CALL OBER_SORT(isp(ielfla_kno),isp(ielfla_ele),isp(ielfla_adr),
     *               nelfla,nelfla_max,nkd_obe,
     *               elpar,isp(ipermut),
     *               isp(ihelp_kno),isp(ihelp_ele),isp(ihelp_num))


      CALL OBER_PRINT(coord,coord_num,
     *                isp(ielfla_adr),isp(ielfla_kno),
     *                nelfla,nelfla_max,nkd_obe,
     *                farb_geb,farb_adr,nfarb_geb,farb_per,
     *                isp(igeb1_help),isp(igeb2_help),
     *                isp(igeb3_help),isp(igeb4_help),
     *                isp(iint1_help),isp(iint2_help),
     *                ober_name_geo,ober_name_ses)

      CALL DEALLOC_ALLE(ifine,lmax)
c     *****************************************************************

      speich_max_sub=speich_max

      return
      end

