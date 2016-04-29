C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_2DE(lnods,elfla_kno,elfla_ele,nelfla,
     *                    nelfla_max,nkd_obe,
     *                    kern_ele,kern_adr,coord_num)
  
      implicit none     

      include 'common.zer'

      integer lnods,elfla_kno,elfla_ele,nelfla,nelfla_max,nkd_obe,
     *        kern_ele,kern_adr,
     *        coord_num

      integer i,j,ielem

      dimension lnods(nelem_max,nkd),
     *          elfla_kno(nelfla_max,nkd_obe),elfla_ele(nelfla_max),
     *          kern_ele(nelem),kern_adr(ngebiet+1),
     *          coord_num(npoin_max)
c     *****************************************************************


c     *****************************************************************
c     SCHREIBEN DER 2-D KERNELEMENTE:                        

      do 142 i=1,nelfla_max
        elfla_ele(i)=0
        do 143 j=1,nrbknie
           elfla_kno(i,j)=0
 143     continue
 142  continue

      do 150 i=1,nelem  
         ielem=kern_ele(i)
         elfla_ele(i)=ielem
         do 151 j=1,nkd
            elfla_kno(i,j)=lnods(ielem,j)
 151     continue
 150  continue
c     *****************************************************************
      return
      end

