"""
Originally written by Antoine Mahoro for combining dithered NIRWALS observations. 

Maps irregularly-placed fiber spectra onto a regular spatial grid (datacube).
The Gaussian-kernel ("modified Shepard's method") reconstruction follows the MaNGA DRP approach (Law+2016)
"""
import numpy as np
from copy import copy
from progress.bar import FillingCirclesBar

def change_coords(core_x_pixels,core_y_pixels,core_diameter_pixels,
                  input_grid_dims,output_grid_dims):
    '''
    This tool is used to map the fiber core centroid locations on the image plane to a new grid covering the same field of view (FOV) but with an arbitrary pixel scale. This pixel scale is set by the ratio of `input_grid_dims` to `output_grid_dims`. Consequently, the coordinates of objects in an input grid with dimensions (10,10) could be mapped to (3,3) with a scale of 0.333.
    
    Output FOV is always the same as the input FOV! The scale is the only thing that differs.The scale must be equal in both x- and y- dimensions. Consequently, an initial grid of (10,15) cannot be scaled to (5,3) as this would require a scale of 2 in the y-direction and a scale of 5 in the x-direction. The output `core_x_pixels` and `core_y_pixels` will have a coordinate system defined with with (0,0) at the upper left corner of the image. Therefore, with output grid dimensions of (10,10), the center of the fiber array would be (5,5).
    
    Argument descriptions are the same as for the ifu_observe function.
    
    KEYWORDS:
    
    input_grid_dims: (int,tuple) the dimensions of the input grid in which the coordinates of the fiber cores are set.
    
     output_grid_dims: (int,tuple) the dimensions of the output grid into which the coordinates of the fiber cores are to be determined.
    
    RETURNS:
    
    core_x_pixels AND core_y_pixels AND core_diameter_pixels: (ndarrays) scaled to the new grid dimensions.
    '''
    
    if type(input_grid_dims)==tuple and type(output_grid_dims)==tuple:
        try:
            osize_y,osize_x = output_grid_dims[0],output_grid_dims[1]
            size_y,size_x = input_grid_dims[0],input_grid_dims[1]
        except:
            raise Exception('Grid dimensions are tuples but do not contain two elements. Stopping...')
    elif type(input_grid_dims)==int and type(output_grid_dims)==int:
        osize_y,osize_x = output_grid_dims,output_grid_dims
        size_y,size_x = input_grid_dims,input_grid_dims
    else:
        raise Exception('Grid dimensions must either be both tuples, e.g. (nrows,ncols), or both ints. Stopping...')
        
    if type(core_x_pixels) in [float,int]: 
        core_x_pixels = np.array([core_x_pixels,]).astype(float)
    elif type(core_x_pixels) in [list,np.ndarray]:
        core_x_pixels = np.array(core_x_pixels).astype(float)
    else:
        try: 
            core_x_pixels = np.array([float(core_x_pixels),])
        except:
            raise Exception("core_x_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_y_pixels) in [float,int]:
        core_y_pixels = np.array([core_y_pixels,]).astype(float)
    elif type(core_y_pixels) in [list,np.ndarray]:
        core_y_pixels = np.array(core_y_pixels).astype(float)
    else:
        try: 
            core_y_pixels = np.array([float(core_y_pixels),])
        except:
            raise Exception("core_y_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_diameter_pixels) in [float,int]:
        core_diameter_pixels = np.array([core_diameter_pixels,]).astype(float)
    elif type(core_diameter_pixels) in [list,np.ndarray]:
        core_diameter_pixels = np.array(core_diameter_pixels).astype(float)
    else:
        try: 
            core_diameter_pixels = np.array([float(core_diameter_pixels),])
        except:
            raise Exception("core_diameter_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    # check that x,y core position array dimensions match
    if core_x_pixels.shape != core_y_pixels.shape:
        raise Exception("Fiber core x- and y- position arrays (or lists/values) do not have matching dimensions. Stopping...")
    
    N_fibers = core_x_pixels.shape[0]
    # core radius not necessarily constant but may be particular to each fiber
    if core_diameter_pixels.shape[0] not in [1,N_fibers]:
        raise Exception("Fiber core_diameter_pixels must either be a single float (all/any fibers have the same diameter) or an array/list of length equal to `core_x_pixels` and `core_y_pixels`. Stopping...")
    
    scale = float(osize_y)/size_y
    scale_x = float(osize_x)/size_x
    if scale_x != scale:
        raise Exception("The scale by which the input coordinates are converted to output coordinates must be the same in x- and y- dimensions. Make sure that `input_grid_dims` and `output_grid_dims` satisfy this condition. Stopping...")
    
    core_x_pixels *= scale
    core_y_pixels *= scale
    core_diameter_pixels *= scale
    
    return(core_x_pixels.flatten(),
           core_y_pixels.flatten(),
           core_diameter_pixels.flatten())

def ifu_to_grid(fiber_data,core_x_pixels,core_y_pixels,core_diameter_pixels,
                grid_dimensions_pixels,use_gaussian_weights=True,
                gaussian_sigma_pixels=1.4, rlim_pixels=3.2,
                use_broadcasting=False,ivar_data=None):
    '''
    Overview: 
    
    With a fiber core array [ndarray] with shape (`N_fibers`,`Nels`) along with their x- and y- coordinates (centroid) on a grid of dimensions `grid_dimensions_pixels`, this tool computes the intensity contribution of each fiber to each pixel of the grid. 
    There are two options:
    (1) The intensity is distributed uniformly over all pixels completely contained within the fiber and partially within pixels that partially overlap with the fiber. In this respect, the method is exactly analogous to ifu_observe but in reverse. Pixels that partially overlap with the fiber receive a portion of the intensity that is weighted by fraction of area of overlap with respect to a full pixel size. This method conserves intensity. This is checked by taking the sum along axis=0 of `fiber_data` (the fiber axis) and comparing to the sum in each wavelength/losvd slice (axis=(1,2)) of the output datacube. The output cube will therefore have a total sum that is equal to the sum of fiber_data, but distributed on the grid. 
    (2) The weights are determined by adopting a gaussian distribution of the intensity from each fiber on the output grid. This method emulates the SDSS-IV MaNGA data reduction pipeline. Specifically, see LAW et al. 2016 (AJ,152,83), Section 9.1 on the construction of the regularly sampled cubes from the non-regularly sampled fiber data. The intensity contribution from each fiber, f[i], to each regularly spaced output pixel is mapped by a gaussian distribution: w[i,j] = exp( -0.5 * r[i,j]^2 / sigma^2 )
    Where r[i,j] is the distance between the fiber core and the pixel centroid and sigma is a constant of decay (taken to by 0.7 arcsec for MaNGA, for example). These weights are necessarily normalized to conserve intensity: W[i,j] = w[i,j] / SUM(w[i,j]) from k=1 to N_fibers 
    Where the sum is over all fibers. Note that there is a distinction between the N_fibers used in LAW et al. 2016 and the one used here. Here, N_fibers refers to all fibers from all exposures (equivalent to the N used by Law et al. 2016). Additionally, the `alpha` parameter used in LAW et al. 2016 is computed and applied to the intensities. `alpha` converts the "per fiber" intensity to the "per pixel" intensity in the new grid. The resulting `out_cube` conserves intensity from the original cube in the limit where there is adequate sampling of the original cube by the fibers. With sparse sampling, the intensity is not necessarily conserved. Note that this differes from (1) which only conserves intensities within the fibers themselves. Method (2) also allows a scale, `rlim_pixels` beyond which a fiber contributes no intensity in the output grid (weights are zero). 
    
    params:
    
    fiber_data: (np.ndarray) with shape (N_fibers,Nels) are the fiber data arrays (core array). These contain the spectra measured in each fiber to be distributed on the output grid.
    core_x[y]_pixels: (np.array) with shape (N_fibers,) are the x[y] centroid positions of each fiber core in the output grid.
    core_diameter_pixels: (float,np.array) of fiber core diameters in output grid pixels.
    grid_dimensions_pixels: (int,tuple) dimensions of the output grid. If tuple, should be the (spatial_x, spatial_y) shape of the output grid.
    use_gaussian_weights: (boolean) if False (default), use method (1) outlined above. If True, use method (2). If True, the `gaussian_sigma_pixels` and `rlim_pixels` are used to determine the profile of the gaussian distribution used in the weights.
    gaussian_sigma_pixels: (int,float,list,np.ndarray) characteristic size of the 2d circular gaussian used when `use_gaussian_weights` is True. 
    rlim_pixels: (int,float,list,np.ndarray): distance in pixels from a fiber core beyond which the weights assigned to all pixels in a weight map are zero (default None, i.e. the weights extend to infinity). 
    use_broadcasting: (boolean) broadcasting of the weight maps with the spectra from each fiber to produce the output datacubes can be very memory-intensive. You can estimate the memory demand by computing (N_fibers*Nels*output_spatial_y*output_spatial_x*64/8/1e9) for the size of the object that needs to be summed over the N_fibers dimension in Gigabytes. If this exceeds your memory requirements, `use_broadcasting` should be set to False. This will greatly increase the computation time at the expense of memory intensiveness.
    ivar_data: (np.ndarray) of inverse variance data with the same shape as `fiber_data`. Contains the inverse variances for `fiber_data`. If not None, an inverse variance cube will be produced alongside the output flux cube with the same shape.
    '''
    fiber_data = np.array(fiber_data,dtype=float)
            
    data_shape = fiber_data.shape
    if len(data_shape) == 1:
        N_fibers,Nels = 1,data_shape[0]
        fiber_data = fiber_data.reshape(1,Nels)
    elif len(data_shape) == 2:
        N_fibers,Nels = data_shape[0],data_shape[1]
    else:
        raise Exception("fiber_data can have either one or two axes. No more, no less. Stopping...")
    
    if ivar_data is not None:
        if ivar_data.shape==data_shape:
            ivar_data.reshape(N_fibers,Nels)
        else:
            raise Exception("ivar_data must have same shape as fiber_data. Stopping...")
    
    if type(core_x_pixels) in [float,int]: 
        core_x_pixels = np.array([core_x_pixels,]).astype(float)
    elif type(core_x_pixels) in [list,np.ndarray]:
        core_x_pixels = np.array(core_x_pixels).astype(float)
    else:
        try: 
            core_x_pixels = np.array([float(core_x_pixels),])
        except:
            raise Exception("core_x_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_y_pixels) in [float,int]:
        core_y_pixels = np.array([core_y_pixels,]).astype(float)
    elif type(core_y_pixels) in [list,np.ndarray]:
        core_y_pixels = np.array(core_y_pixels).astype(float)
    else:
        try: 
            core_y_pixels = np.array([float(core_y_pixels),])
        except:
            raise Exception("core_y_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_diameter_pixels) in [float,int]:
        core_diameter_pixels = np.array([core_diameter_pixels,]).astype(float)
    elif type(core_diameter_pixels) in [list,np.ndarray]:
        core_diameter_pixels = np.array(core_diameter_pixels).astype(float)
    else:
        try: 
            core_diameter_pixels = np.array([float(core_diameter_pixels),])
        except:
            raise Exception("core_diameter_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    # check that x,y core position array dimensions match
    if core_x_pixels.shape != core_y_pixels.shape:
        raise Exception("Fiber core x- and y- position arrays (or lists/values) do not have matching dimensions. Stopping...")
    
    # core radius not necessarily constant but may be particular to each fiber
    if core_diameter_pixels.shape[0] not in [1,N_fibers]:
        raise Exception("Fiber core_diameter_pixels must either be a single float (all/any fibers have the same diameter) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")
    core_radius_pixels = core_diameter_pixels/2
    core_radius_pixels = core_radius_pixels.reshape(-1,1,1)
    core_x_pixels = core_x_pixels.reshape(-1,1,1)
    core_y_pixels = core_y_pixels.reshape(-1,1,1)

    if type(grid_dimensions_pixels) == int:
        size_y,size_x = grid_dimensions_pixels,grid_dimensions_pixels
    elif type(grid_dimensions_pixels) == tuple:
        size_y = int(grid_dimensions_pixels[1])
        size_x = int(grid_dimensions_pixels[0])
    else:
        raise Exception("grid_dimensions_pixels must be an int or a tuple of two ints (e.g. '(10,10)')")
        
    Y,X = np.ogrid[0:size_y,0:size_x]
    Y = Y[np.newaxis,...]
    X = X[np.newaxis,...]
    
    if not use_gaussian_weights:
        #############################
        # Inverse Drizzle Algorithm #
        #############################
        
        # initialize weight map
        weight_map = np.zeros((N_fibers,size_y,size_x)).astype(int)

        # select rectangular region around fiber to refine for weight estimates
        weight_map[(np.abs(X+0.5-core_x_pixels)<core_radius_pixels+0.5) * 
                   (np.abs(Y+0.5-core_y_pixels)<core_radius_pixels+0.5)] = 1

        indices = np.argwhere(weight_map)
        slices,rows,cols = indices[:,0],indices[:,1],indices[:,2]

        row_min = [np.min(rows[slices==i]) for i in range(N_fibers)]
        row_max = [np.max(rows[slices==i])+1 for i in range(N_fibers)]
        col_min = [np.min(cols[slices==i]) for i in range(N_fibers)]
        col_max = [np.max(cols[slices==i])+1 for i in range(N_fibers)]

        # the refined grid is defined to have a minimum of 100 pixels
        # across the diameter of the fiber
        rfactor = (100/core_diameter_pixels).astype(int)
        # if the diameter exceeds 100 pixels already, 
        # use the original regular grid
        rfactor[rfactor<1]=1    

        # handling condition where a single core diameter is given
        if len(rfactor)==1 and N_fibers>1:
            core_radius_pixels = np.ones((N_fibers),dtype=int).flatten()*core_radius_pixels.flatten()
            rfactor = np.ones((N_fibers),dtype=int)*rfactor.flatten()
        weight_map = weight_map.astype(float)
        
        # Each refined patch is unique to the scale factor.
        # Each original patch can also differ. Therefore, need loop.
        for i in np.arange(N_fibers):

            col_max_ = col_max[i]
            col_min_ = col_min[i]
            row_max_ = row_max[i]
            row_min_ = row_min[i]
            rfactor_ = rfactor[i]
            core_x_pixels_ = core_x_pixels[i]
            core_y_pixels_ = core_y_pixels[i]
            core_radius_pixels_ = core_radius_pixels[i]

            rsize_x = (col_max_-col_min_)*rfactor_
            rsize_y = (row_max_-row_min_)*rfactor_
            Yr,Xr = np.ogrid[0:rsize_y,0:rsize_x]
            maskr = np.zeros((rsize_y,rsize_x))

            maskr[np.sqrt((Xr+col_min_*rfactor_+0.5 
                           - core_x_pixels_*rfactor_)**2 
                          +(Yr+row_min_*rfactor_+0.5 
                            - core_y_pixels_*rfactor_)**2) 
                  < core_radius_pixels_*rfactor_ ]=1
            
            weight_mapr = np.zeros((size_y*rfactor_,size_x*rfactor_))
            weight_mapr[row_min_*rfactor_:row_max_*rfactor_,
                        col_min_*rfactor_:col_max_*rfactor_] = maskr
            patch = maskr.reshape(row_max_-row_min_,rfactor_,
                                  col_max_-col_min_,rfactor_)
            patch = patch.sum(axis=(1,3)).astype(float)/rfactor_**2
            weight_map[i,row_min_:row_max_,col_min_:col_max_]=patch
            
        # spatial flux conservation term
        alpha = 1./np.nansum(weight_map,axis=(1,2)).reshape(-1,1,1)

        # slow but non-memory intensive
        if not use_broadcasting:
            # perform reconstruction channel-by-channel
            out_cube = np.zeros((Nels,size_y,size_x))
            if ivar_data is not None:
                ivar_cube = np.zeros_like(out_cube)
            print()
            bar = FillingCirclesBar('Spatial reconstruction', max=Nels)
            for i in range(Nels):
                weight_map_el = copy(weight_map)
                weight_map_el[np.isnan(fiber_data[:,i])] = np.nan
                normalization_el = np.nansum(weight_map_el,axis=0)
                normalization_el[normalization_el==0]=np.nan
                weight_map_el/=normalization_el
                weight_map_el*=alpha
                weight_map_el[np.isnan(weight_map)]=0.
                out_el = np.nansum(fiber_data[:,i].reshape(N_fibers,1,1)
                                   *weight_map_el,axis=0)
                out_el[np.isnan(out_el)]=0.
                out_cube[i] = out_el
                if ivar_data is not None:
                    ivar_el = 1./np.nansum(weight_map_el**2/ivar_data[:,i].reshape(N_fibers,1,1),axis=0)
                    ivar_el[np.isnan(ivar_el)]=0.
                    ivar_cube[i] = ivar_el
                bar.next()
            bar.finish()
            print()
            
        # fast but highly memory intensive
        else:
            weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)
            weight_map = weight_map*np.ones((1,Nels,1,1))
            weight_map[np.isnan(fiber_data),:,:] = np.nan      
            normalization = np.nansum(weight_map,axis=0)
            normalization[normalization==0]=np.nan
            weight_map/=normalization
            weight_map*=alpha.reshape(N_fibers,1,1,1)
            weight_map[np.isnan(weight_map)]=0.
            out_cube = np.nansum(fiber_data.reshape(N_fibers,Nels,1,1)
                                 *weight_map,axis=0)
            out_cube[np.isnan(out_cube)]=0.
            if ivar_data is not None:
                ivar_cube = 1./np.nansum(weight_map**2/ivar_data.reshape(N_fibers,Nels,1,1),axis=0)
                ivar_cube[np.isnan(ivar_cube)]=0.
            
        if ivar_data is not None:    
            return [out_cube,ivar_cube],weight_map
        else:
            return out_cube,weight_map
    
    else:
        ##############################
        # Modified Shepard Algorithm #
        ##############################
        
        # check `gaussian_sigma_pixels` type
        if type(gaussian_sigma_pixels) in [float,int]:
            gaussian_sigma_pixels = \
            np.array([gaussian_sigma_pixels,]).astype(float)
        elif type(gaussian_sigma_pixels) in [list,np.ndarray]:
            gaussian_sigma_pixels = \
            np.array(gaussian_sigma_pixels).astype(float)
        else:
            try: 
                gaussian_sigma_pixels = np.array([float(gaussian_sigma_pixels),])
            except:
                raise Exception("`gaussian_sigma_pixels` not in accepted format. Use a list, numpy array, int, or float. Stopping...")

        # gaussian_sigma_pixels not necessarily constant 
        if gaussian_sigma_pixels.shape[0] not in [1,N_fibers]:
            raise Exception("`gaussian_sigma_pixels` must either be a single float (all/any fibers have the same sigma) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")

        # handling condition where a gaussian_sigma_pixels is given
        if len(gaussian_sigma_pixels)==1 and N_fibers>1:
            gaussian_sigma_pixels = np.ones((N_fibers),dtype=int).flatten()*gaussian_sigma_pixels.flatten()
        
        # generate cube of 2d gaussian pdfs with output grid dimensions
        r2 = ((X+0.5)-core_x_pixels)**2 + ((Y+0.5)-core_y_pixels)**2
        weight_map = np.exp(-0.5*r2/gaussian_sigma_pixels.reshape(N_fibers,1,1)**2)
        
        # for each fiber, all pixel weights outside `rlim_pixels` are zero
        if rlim_pixels is not None:
            
            if type(rlim_pixels) in [float,int]:
                rlim_pixels = np.array([rlim_pixels,]).astype(float)
            elif type(rlim_pixels) in [list,np.ndarray]:
                rlim_pixels = np.array(rlim_pixels).astype(float)
            else:
                try: 
                    rlim_pixels = np.array([float(rlim_pixels),])
                except:
                    raise Exception("`rlim_pixels` not in accepted format. Use a list, numpy array, int, or float. Stopping...")

            # gaussian_sigma_pixels not necessarily constant but may be particular to each fiber
            if rlim_pixels.shape[0] not in [1,N_fibers]:
                raise Exception("`rlim_pixels` must either be a single float (all/any fibers have the same sigma) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")

            # handling condition where a gaussian_sigma_pixels is given
            if len(rlim_pixels)==1 and N_fibers>1:
                rlim_pixels = np.ones((N_fibers),dtype=int).flatten()*rlim_pixels.flatten()
        
            weight_map[np.sqrt(r2)>rlim_pixels.reshape(N_fibers,1,1)] = 0
        
        # normalization to determine intensity contribution of each fiber to each pixel as in LAW et al 2016
        
        # spatial flux conservation term
        alpha = 1./(np.pi*(core_radius_pixels)**2)
        
        # slow but non-memory intensive
        if not use_broadcasting:
            # perform reconstruction channel-by-channel
            out_cube = np.zeros((Nels,size_y,size_x))
            if ivar_data is not None:
                ivar_cube = np.zeros_like(out_cube)
            print()
            bar = FillingCirclesBar('Spatial reconstruction', max=Nels)
            for i in range(Nels):
                weight_map_el = copy(weight_map)
                weight_map_el[np.isnan(fiber_data[:,i])] = np.nan
                normalization_el = np.nansum(weight_map_el,axis=0)
                normalization_el[normalization_el==0]=np.nan
                weight_map_el/=normalization_el
                weight_map_el*=alpha
                weight_map_el[np.isnan(weight_map)]=0.
                out_el = np.nansum(fiber_data[:,i].reshape(N_fibers,1,1)
                                   *weight_map_el,axis=0)
                out_el[np.isnan(out_el)]=0.
                out_cube[i] = out_el
                if ivar_data is not None:
                    ivar_el = 1./np.nansum(weight_map_el**2/ivar_data[:,i].reshape(N_fibers,1,1),axis=0)
                    ivar_el[np.isnan(ivar_el)]=0.
                    ivar_cube[i] = ivar_el
                bar.next()
            bar.finish()
            print()
            
        # fast but highly memory intensive
        else:
            weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)
            weight_map = weight_map*np.ones((1,Nels,1,1))
            weight_map[np.isnan(fiber_data),:,:] = np.nan      
            normalization = np.nansum(weight_map,axis=0)
            normalization[normalization==0]=np.nan
            weight_map/=normalization
            if alpha.size==1:
                alpha = np.ones(N_fibers).reshape(-1,1,1,1)*alpha
            weight_map*=alpha
            weight_map[np.isnan(weight_map)]=0.
            out_cube = np.nansum(fiber_data.reshape(N_fibers,Nels,1,1)
                                 *weight_map,axis=0)
            out_cube[np.isnan(out_cube)]=0.
            if ivar_data is not None:
                ivar_cube = 1./np.nansum(weight_map**2/ivar_data.reshape(N_fibers,Nels,1,1),axis=0)
                ivar_cube[np.isnan(ivar_cube)]=0.
        
        if ivar_data is not None:    
            return [out_cube,ivar_cube],weight_map
        else:
            return out_cube,weight_map