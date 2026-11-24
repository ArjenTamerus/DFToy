program DFToy
  use parameters
  !use parallel
  use iterative

  implicit none

  real(kind=dp), dimension(:,:,:), allocatable :: H_kinetic, &
                                                   H_local

  integer :: npw_1d
  !type(dftoy_params) :: params

  ! todo get params
  ! call get_config()
  call set_default_config()

  npw_1d = 2*params%num_wavevectors + 1

  allocate(H_kinetic(npw_1d, npw_1d, npw_1d))
  allocate(H_local(npw_1d, npw_1d, npw_1d))

  call init_kinetic(H_kinetic)
  call init_local(H_local)

  call iterative_solver(H_kinetic, H_local)

  deallocate(H_local)
  deallocate(H_kinetic)

  call dftoy_print("Done")

  contains

    subroutine init_kinetic(H_kinetic)
      real(kind=dp),dimension(:,:,:),intent(out),target,contiguous :: H_kinetic

      real(kind=dp),dimension(:,:,:), pointer :: H_shift
      integer :: x, y, z
      integer ::num_wv

      num_wv = params%num_wavevectors

      H_shift(-num_wv:num_wv,-num_wv:num_wv,-num_wv:num_wv) => H_kinetic

      do z=-num_wv,num_wv
        do y=-num_wv,num_wv
          do x=-num_wv,num_wv
            H_shift(x,y,z) = 0.5_dp*real(x**2+y**2+z**2,kind=dp)
          end do
        end do
      end do
    end subroutine

    subroutine init_local(H_local)
      real(kind=dp),dimension(:,:,:),intent(out),target,contiguous :: H_local

      real(kind=dp),dimension(:,:,:), pointer :: H_shift
      integer :: x, y, z
      integer ::num_wv

      num_wv = params%num_wavevectors

      H_shift(-num_wv:num_wv,-num_wv:num_wv,-num_wv:num_wv) => H_local

      do z=-num_wv,num_wv
        do y=-num_wv,num_wv
          do x=-num_wv,num_wv

            H_shift(x,y,z) = -0.37_dp/(0.005_dp+abs(sqrt(real(x*x+y*y+z*z))/(num_wv*num_wv*num_wv) - 0.5_dp))
          end do
        end do
      end do
    end subroutine
end program
