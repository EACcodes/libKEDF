#ifdef LIBKEDF_ENABLE
module libkedfinterface
    
    use iso_c_binding
        
    implicit none
    
    type(c_ptr), save :: libkedf_data
    logical, save :: uselibkedf = .false.
    logical, save :: usegpucode = .false.
    logical, save :: usesingleprec = .false.
    
    interface
        function libkedf_init ()
            
            use iso_c_binding

            type(c_ptr) :: libkedf_init
            
        end function libkedf_init
    end interface
    
    interface
        subroutine libkedf_initialize_grid (libkedf_data, x, y, z, vecX, vecY, vecZ)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            integer(kind=c_int), intent(in) :: x,y,z
            real(kind=8), intent(in) :: vecX, vecY, vecZ
            
        end subroutine libkedf_initialize_grid
    end interface
    
#ifdef LIBKEDF_OCL
    interface
        subroutine libkedf_initialize_grid_ocl (libkedf_data, x, y, z, vecX, vecY, vecZ, platformNo, deviceNo)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            integer(kind=c_int), intent(in) :: x,y,z, platformNo, deviceNo
            real(kind=8), intent(in) :: vecX, vecY, vecZ
            
        end subroutine libkedf_initialize_grid_ocl
    end interface
#endif
    
    interface
        subroutine libkedf_update_cellvectors (libkedf_data, vecX, vecY, vecZ)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=8), intent(in) :: vecX, vecY, vecZ
            
        end subroutine libkedf_update_cellvectors
    end interface

    interface
        subroutine libkedf_initialize_tf (libkedf_data)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            
        end subroutine libkedf_initialize_tf
    end interface
    
    interface
        subroutine libkedf_initialize_vw (libkedf_data)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            
        end subroutine libkedf_initialize_vw
    end interface
    
    interface
        subroutine libkedf_initialize_tf_plus_vw (libkedf_data, a, b)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: a,b
            
        end subroutine libkedf_initialize_tf_plus_vw
    end interface
    
    interface
        subroutine libkedf_initialize_wt (libkedf_data, rho0)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0
            
        end subroutine libkedf_initialize_wt
    end interface
    
    interface
        subroutine libkedf_initialize_sm (libkedf_data, rho0)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0
            
        end subroutine libkedf_initialize_sm
    end interface
    
    interface
        subroutine libkedf_initialize_wgc1st (libkedf_data, rho0)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0
            
        end subroutine libkedf_initialize_wgc1st
    end interface
    
    interface
        subroutine libkedf_initialize_wgc2nd (libkedf_data, rho0)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0
            
        end subroutine libkedf_initialize_wgc2nd
    end interface

    interface
        subroutine libkedf_initialize_wgcfull (libkedf_data, rho0)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0
            
        end subroutine libkedf_initialize_wgcfull
    end interface
    
    interface
        subroutine libkedf_initialize_hc (libkedf_data, rho0, lambda)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=c_double), intent(in) :: rho0, lambda
            
        end subroutine libkedf_initialize_hc
    end interface
    
    interface
        subroutine libkedf_energy (libkedf_data, grid, energy)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=8), intent(in) :: grid
            real(kind=8), intent(out) :: energy
            
        end subroutine libkedf_energy
    end interface
    
    interface
        subroutine libkedf_potential (libkedf_data, grid, potential, energy)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            real(kind=8), intent(in) :: grid
            real(kind=8), intent(inout) :: potential
            real(kind=8), intent(out) :: energy
            
        end subroutine libkedf_potential
    end interface
    
    interface
        subroutine libkedf_cleanup (libkedf_data)
            
            use iso_c_binding

            type(c_ptr), intent(inout) :: libkedf_data
            
        end subroutine libkedf_cleanup
    end interface
    
end module libkedfinterface
#endif
