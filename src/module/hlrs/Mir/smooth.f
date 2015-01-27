!=======================================================================
!       This module is for smoothing a value across
!       the whole mesh. The idea is simply to
!       accumulate inner values at the nodes of a cell
!       and redistribute the node values to the inner value.
!
!	Original author: Michael M. Resch (MR)
!	First released: 1.November 1996
!	Changed: 7.3.1998
!		Include smoothing of corner cell values
!		subroutine smooth_corners
!
!=======================================================================
	module visco_smooth
	contains

!=======================================================================
	subroutine visco_init_smoother()
!=======================================================================
!	this subroutine initialises an operator for
!       the smoothing of a value.
!=======================================================================
	use visco_declaration
	use visco_control

	implicit none

	integer :: i
        integer,dimension(8) :: n

!=======================================================================
!       calculate sums of neighbouring volumes.
!       S_1 = ((i,j)     + (i-1,j)   + (i,j-1)   + (i-1,j-1))^-1
!       S_2 = ((i+1,j)   + (i,j)     + (i+1,j-1) + (i,j-1))^-1
!       S_3 = ((i,j+1)   + (i-1,j+1) + (i,j)     + (i-1,j))^-1
!       S_4 = ((i+1,j+1) + (i,j+1)   + (i+1,j)   + (i,j))^-1
!=======================================================================
        do i = 1,dim
	  n(1) = cell_data(i)%neighbours(1)
	  n(2) = cell_data(i)%neighbours(2)
	  n(3) = cell_data(i)%neighbours(3)
	  n(4) = cell_data(i)%neighbours(4)
          if (n(1) .ne. -1) then
	    n(5) = cell_data(n(1))%neighbours(2)
          else
            n(5) = cell_data(n(2))%neighbours(1)
          end if
          if (n(3) .ne. -1) then
  	    n(6) = cell_data(n(3))%neighbours(2)
          else
	    n(6) = cell_data(n(2))%neighbours(3)
	  end if
          if (n(3) .ne. -1) then
	    n(7) = cell_data(n(3))%neighbours(4)
	  else
	    n(7) = cell_data(n(4))%neighbours(3)
	  end if
          if (n(1) .ne. -1) then
	    n(8) = cell_data(n(1))%neighbours(4)
          else
	    n(8) = cell_data(n(4))%neighbours(1)
	  end if
          
          cell_data(i)%sum(1) = 1.0d0/(cell_data(i)%volume    +	&
				cell_data(n(3))%volume +	&
				cell_data(n(4))%volume +	&
				cell_data(n(7))%volume)
          cell_data(i)%sum(2) = 1.0d0/(cell_data(i)%volume    +	&
				cell_data(n(1))%volume +	&
				cell_data(n(4))%volume +	&
				cell_data(n(8))%volume)
          cell_data(i)%sum(3) = 1.0d0/(cell_data(i)%volume    +	&
				cell_data(n(2))%volume +	&
				cell_data(n(3))%volume +	&
				cell_data(n(6))%volume)
          cell_data(i)%sum(4) = 1.0d0/(cell_data(i)%volume    +	&
				cell_data(n(1))%volume +	&
				cell_data(n(2))%volume +	&
				cell_data(n(5))%volume)
        end do

	end subroutine visco_init_smoother

!=======================================================================
	subroutine visco_smoother()
!=======================================================================
!	this subroutine smooths the pressure using
!       sums calculated in the init_smoother.
!       actually this routine smooths by first puuting all
!       values from the inner point to the corner points (taking
!       into account the size of each volume) and then calculating
!       a new inner value from the corner values.
!=======================================================================
	use visco_declaration
	use visco_control

	implicit none

	integer :: i
	integer, dimension(8)      :: n
	real(double), dimension(8) :: volume,sum

        do i = 1,dim
	  n(1) = cell_data(i)%neighbours(1)
	  n(2) = cell_data(i)%neighbours(2)
	  n(3) = cell_data(i)%neighbours(3)
	  n(4) = cell_data(i)%neighbours(4)
          if (n(1) .ne. -1) then
	    n(5) = cell_data(n(1))%neighbours(2)
          else
            n(5) = cell_data(n(2))%neighbours(1)
          end if
          if (n(3) .ne. -1) then
  	    n(6) = cell_data(n(3))%neighbours(2)
          else
	    n(6) = cell_data(n(2))%neighbours(3)
	  end if
          if (n(3) .ne. -1) then
	    n(7) = cell_data(n(3))%neighbours(4)
	  else
	    n(7) = cell_data(n(4))%neighbours(3)
	  end if
          if (n(1) .ne. -1) then
	    n(8) = cell_data(n(1))%neighbours(4)
          else
	    n(8) = cell_data(n(4))%neighbours(1)
	  end if


	  volume(1) = cell_data(i)%volume
	  volume(2) = cell_data(i)%volume
	  volume(3) = cell_data(i)%volume
	  volume(4) = cell_data(i)%volume
          if (n(1) .ne. -1) then
	    volume(5) = cell_data(n(1))%volume
          else
            volume(5) = cell_data(n(2))%volume
          end if
          if (n(3) .ne. -1) then
  	    volume(6) = cell_data(n(3))%volume
          else
	    volume(6) = cell_data(n(2))%volume
	  end if
          if (n(3) .ne. -1) then
	    volume(7) = cell_data(n(3))%volume
	  else
	    volume(7) = cell_data(n(4))%volume
	  end if
          if (n(1) .ne. -1) then
	    volume(8) = cell_data(n(1))%volume
          else
	    volume(8) = cell_data(n(4))%volume
	  end if

	  sum(1) = cell_data(i)%sum(1)
	  sum(2) = cell_data(i)%sum(2)
	  sum(3) = cell_data(i)%sum(3)
	  sum(4) = cell_data(i)%sum(4)


          t(i) = (u(i)*cell_data(i)%volume*(sum(1)+sum(2)+sum(3)+sum(4)) + &
                  u(n(1))*cell_data(n(1))%volume*(sum(2)+sum(4))	 + &
                  u(n(2))*cell_data(n(2))%volume*(sum(3)+sum(4))	 + &
                  u(n(3))*cell_data(n(3))%volume*(sum(1)+sum(3))         + &
                  u(n(4))*cell_data(n(4))%volume*(sum(1)+sum(2))         + &
                  u(n(5))*cell_data(n(5))%volume*sum(4)                  + &
                  u(n(6))*cell_data(n(6))%volume*sum(3)                  + &
                  u(n(7))*cell_data(n(7))%volume*sum(1)                  + &
                  u(n(8))*cell_data(n(8))%volume*sum(2))*0.25d0
                  
        end do

        end subroutine visco_smoother

!=======================================================================
	subroutine smooth_corners
!=======================================================================
!	This subroutine smooths the corner values. It is
!	well known that sharp corners in a flow field
!	yield physically wrong results at the corner points.
!	Such singularities are due to mathematical
!	singularities at those points. The may be avoided
!	by smoothing out the values. Some people use
!	sort of supercells that simply reduce the extent
!	to which other cells are effected (this is a sort
!	of isolation of the problem)
!
!	Original author: Michael M. Resch
!	First released: 7.3.1998
!	
!=======================================================================
	use visco_declaration

	integer	:: i,j,k,l,count,n

!=======================================================================
	do i = 1,num_corners
	  j = corners(i)
	  count = 0
	  do l = 1,5
            q((j-1)*5+l) = 0.0d0
	  end do
	  do k = 1,4
	    n = cell_data(j)%neighbours(k)
	    if (n .ne. -1) then
	      do l = 1,5
                q((j-1)*5+l) = q((j-1)*5+l)+q((n-1)*5+l)
	      end do
              count = count+1
	    end if
          end do
          do l = 1,5
	    q((j-1)*5+l) = q((j-1)*5+l)/count
	  end do
	end do

	end subroutine smooth_corners

	end module visco_smooth
