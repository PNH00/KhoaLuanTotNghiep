#Nạp các thư viện cần thiết
import numpy as np
import scipy.ndimage as ndi
from skimage.transform import resize


#Tính tích chập 
def compute_gradient(T,filter):
    init_g = np.zeros(T.shape)
    init_g[:,:,0] = ndi.convolve(T[:,:,0], filter,mode='wrap')
    init_g[:,:,1] = ndi.convolve(T[:,:,1], filter,mode='wrap')
    init_g[:,:,2] = ndi.convolve(T[:,:,2], filter,mode='wrap')
    return init_g

def SIRR(I,lamb):
    I = I / 255
    
    #Khởi tạo các biến phụ trợ
    S = np.shape(I)
    size_wh = S[0:2]
    dim_repmat = (S[2], 1,1)
    filter1 = np.array([[1, -1]])
    filter2 = filter1.T
    filter3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    nor1 = np.tile(np.abs(np.fft.fftn(filter3, size_wh)).T, dim_repmat).T**2 * np.fft.fft2(I,axes = (0,1))
    den1 = np.tile(np.abs(np.fft.fftn(filter3,size_wh).T)**2,dim_repmat).T 
    den2 = np.tile(np.abs(np.fft.fftn(filter1,size_wh).T)**2 + np.abs(np.fft.fftn(filter2,size_wh).T)**2,dim_repmat).T

    low_b = np.zeros_like(I)
    g1 = np.zeros_like(I)
    g2 = np.zeros_like(I)
    up_b = I
    eps = 1e-16
    beta = 20
    T = I
    iter_max = 5
    
    while iter_max:
        iter_max-=1
        #Cập nhật g
        g1[:,:,0] = ndi.convolve(T[:,:,0], filter1,mode='wrap')
        g1[:,:,1] = ndi.convolve(T[:,:,1], filter1,mode='wrap')
        g1[:,:,2] = ndi.convolve(T[:,:,2], filter1,mode='wrap')
        mask = np.tile(np.sum(np.abs(g1), axis=2).T < 1/beta, dim_repmat).T
        g1[mask] = 0
    
        g2[:,:,0] = ndi.convolve(T[:,:,0], filter2,mode='wrap')
        g2[:,:,1] = ndi.convolve(T[:,:,1], filter2,mode='wrap')
        g2[:,:,2] = ndi.convolve(T[:,:,2], filter2,mode='wrap')
        mask_2 = np.tile(np.sum(np.abs(g2), axis=2).T < 1/beta, dim_repmat).T
        g2[mask_2] = 0

        #Tính toán T
        nor2 = np.concatenate((np.concatenate([[g1[:, -1, :] - g1[:, 0, :]]], axis=2).reshape(-1, 1,3),-np.diff(g1,axis=1)),axis=1) +\
            np.concatenate((np.concatenate([[g2[-1,:,:] - g2[0,:,:]]], axis=2).reshape(1, -1,3),-np.diff(g2, axis=0)),axis=0)
    
        nor = lamb * nor1 + beta * np.fft.fft2(nor2,axes = (0,1))
    
        den = lamb * den1 + beta * den2 + eps

        T = np.real(np.fft.ifft2(nor/den, axes = (0,1)))

        #Chuẩn hóa, điều chỉnh T cho phù hợp 
        for i in range(dim_repmat[0]):
            max_iteration = 100
            T_t = T[:,:,i]
            threshold = 1/np.size(T_t)
            while max_iteration:
                max_iteration-=1
                dt_nor = np.sum(T_t[T_t < low_b[:, :, i]])

                dt = -2 * (dt_nor + np.sum(T_t[T_t > up_b[:, :, i]])) / np.size(T_t)
      
                T_t = T_t + dt
                if np.abs(dt) < threshold:
                    break
            T[:,:,i] = T_t

        T[T < low_b] = low_b[T < low_b]
        T[T > up_b] = up_b[T > up_b]

        beta *= 2
    R = I - T
    return T,R




def SIRRImprove(I,lamb,first_iter_max = 5,second_max_iteration = 50):
    image = I
    I = resize(I , (I.shape[0] *1.2,I.shape[1] *1.2),anti_aliasing=True)  
    if I.max() > 1:
        I = I / 255
    S = np.shape(I)
    size_wh = S[0:2]
    dim_repmat = (S[2], 1,1)
    filter1 = np.array([[1, -1]])
    filter2 = filter1.T
    filter3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    nor1 = np.tile(np.abs(np.fft.fftn(filter3, size_wh)).T, dim_repmat).T**2 * np.fft.fft2(I,axes = (0,1))
    den1 = np.tile(np.abs(np.fft.fftn(filter3,size_wh).T)**2,dim_repmat).T 
    den2 = np.tile(np.abs(np.fft.fftn(filter1,size_wh).T)**2 + np.abs(np.fft.fftn(filter2,size_wh).T)**2,dim_repmat).T

    low_b = np.zeros_like(I)
    g1 = np.zeros_like(I)
    g2 = np.zeros_like(I)
    up_b = I
    eps = 1e-16
    beta = 20
    LB = I
    iter_max = first_iter_max
    
    while iter_max:
        iter_max-=1
        g1[:,:,0] = ndi.convolve(LB[:,:,0], filter1,mode='wrap')
        g1[:,:,1] = ndi.convolve(LB[:,:,1], filter1,mode='wrap')
        g1[:,:,2] = ndi.convolve(LB[:,:,2], filter1,mode='wrap')
        mask = np.tile(np.sum(np.abs(g1), axis=2).T < 1/beta, dim_repmat).T
        g1[mask] = 0
    
        g2[:,:,0] = ndi.convolve(LB[:,:,0], filter2,mode='wrap')
        g2[:,:,1] = ndi.convolve(LB[:,:,1], filter2,mode='wrap')
        g2[:,:,2] = ndi.convolve(LB[:,:,2], filter2,mode='wrap')
        mask_2 = np.tile(np.sum(np.abs(g2), axis=2).T < 1/beta, dim_repmat).T
        g2[mask_2] = 0

        nor2 = np.concatenate((np.concatenate([[g1[:, -1, :] - g1[:, 0, :]]], axis=2).reshape(-1, 1,3),-np.diff(g1,axis=1)),axis=1) +\
            np.concatenate((np.concatenate([[g2[-1,:,:] - g2[0,:,:]]], axis=2).reshape(1, -1,3),-np.diff(g2, axis=0)),axis=0)
    
        nor = lamb * nor1 + beta * np.fft.fft2(nor2,axes = (0,1))
    
        den = lamb * den1 + beta * den2 + eps

        LB = np.real(np.fft.ifft2(nor/den, axes = (0,1)))

        for i in range(dim_repmat[0]):
            
            max_iteration = second_max_iteration
            LB_t = LB[:,:,i]
            threshold = 1/np.size(LB_t)
            while max_iteration:
                max_iteration-=1
                dt_nor = np.sum(LB_t[LB_t < low_b[:, :, i]])

                dt = -2 * (dt_nor + np.sum(LB_t[LB_t > up_b[:, :, i]])) / np.size(LB_t)
      
                LB_t = LB_t + dt
                if np.abs(dt) < threshold:
                    break
            LB[:,:,i] = LB_t

        LB[LB < low_b] = low_b[LB < low_b]
        LB[LB > up_b] = up_b[LB > up_b]


        beta *= 2
    LR = I - LB
    LR_upsize = resize(LR , (image.shape[0] , image.shape[1] ),anti_aliasing=True)
    LB_upsize = abs(image/255 - LR_upsize)
    return LB_upsize,LR_upsize
    
