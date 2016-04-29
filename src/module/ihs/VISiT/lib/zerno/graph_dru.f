C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GRAPH_DRU(elmat,elmat_adr,elmat_stu,nl_elmat,
     *                     graph_name,graph_all,graph_fla)
c
      implicit none

      include 'common.zer'

      integer   elmat,elmat_adr,elmat_stu,nl_elmat

      integer   i,k,ludru

      logical   graph_all,graph_fla

      character*80 graph_name,comment

      dimension elmat_adr(nelem+1),elmat_stu(nelem),elmat(nl_elmat)
c     ****************************************************************

c     ****************************************************************
c     AUSDRUCK DES GRAPHEN:

      ludru=79 
      open(ludru,file=graph_name,status='unknown')
      if (graph_fla) then
         write(ludru,*)'Flaechen-Graph      '
      else if (graph_all) then
         write(ludru,*)'Kompletter Graph    '
      endif 
      write(ludru,*)'nl_elmat=',nl_elmat  
      write(ludru,*)'                             '
 
      write(ludru,*)' Nr         Graph:'
      do 501 i=1,nelem
        write(ludru,555) i,elmat_adr(i),elmat_stu(i),
     *                     (elmat_adr(i+1)-elmat_adr(i)),
     *                     (elmat_stu(i)+1-elmat_adr(i)),
     *                  (elmat(k),k=elmat_adr(i),elmat_adr(i+1)-1)
 501  continue

      comment='File geschrieben:'
      call char_druck(comment,graph_name,6)
      close(ludru)
 555  format(i5,1x,2(i6,1x),2x,2(i2,1x),2x,30(i5,1x))
c     ===============================================================

      return
      end

