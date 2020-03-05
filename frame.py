from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pdb
import cv2

def load_lib():
    return ctypes.cdll.LoadLibrary('./libframe.so')


def get_histogram(lib, image, synthesized, filters, width, height, im_width, im_height, num_bins, min_response, max_response, is_syn):
    min_response = np.array([min_response])
    max_response = np.array([max_response])
    num_filters = len(filters)
    synthesized = np.copy(synthesized)
    image = np.copy(image)
    response = np.zeros((1, num_filters * num_bins))

    c_array = lambda a: (a.__array_interface__['data'][0] + np.arange(a.shape[0]) * a.strides[0]).astype(np.uintp)
    c_int32 = lambda x: x.astype(np.int32)
    ndpointerpointer = lambda: ndpointer(dtype=np.uintp, ndim=1, flags='C')

    getHistogram = lib.getHistogram2
    getHistogram.restype = None
    getHistogram.argtypes = [ndpointerpointer(), ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ctypes.c_int, ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ctypes.c_int]

    getHistogram(c_array(filters), c_int32(width), c_int32(height), num_filters, c_int32(image), c_int32(synthesized), im_width, im_height, num_bins, response, min_response, max_response, is_syn)

    return np.copy(response), np.copy(min_response[0]), np.copy(max_response[0])


def gibbs(lib, image, synthesized, filtermatrix, width, height, im_width, im_height, num_bins, weights, max_intensity):
    max_size = filtermatrix.shape[1]
    num_filters = filtermatrix.shape[0]
    synthesized = np.copy(synthesized)
    image = np.copy(image)
    weights = np.copy(weights)
    filters = np.copy(filtermatrix)

    syn_response = np.zeros((1, num_filters * num_bins)).astype(np.double)
    c_array = lambda a: (a.__array_interface__['data'][0] + np.arange(a.shape[0]) * a.strides[0]).astype(np.uintp)
    c_int32 = lambda x: x.astype(np.int32)
    c_double = lambda x: x.astype(np.double)
    ndpointerpointer = lambda: ndpointer(dtype=np.uintp, ndim=1, flags='C')

    Gibbs = lib.Gibbs
    Gibbs.restype = None
    Gibbs.argtypes = [ndpointer(ctypes.c_double), ndpointerpointer(), ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int]

    Gibbs(c_double(weights), c_array(filters), c_int32(width), c_int32(height), num_filters, num_bins, c_int32(image), synthesized, syn_response, im_width, im_height, max_intensity)
    
    return np.copy(synthesized), np.copy(syn_response)


def main():
    from skimage import transform, io
    from filters import get_filters
    lib = load_lib()
    np.random.seed(100)
    
    max_intensity = 7
    im_w = im_h = 256

    [F, filters, width, height] = get_filters()
    num_filters = len(filters)  
    synthesized = (max_intensity * np.random.rand(im_w * im_h)).astype(np.int32)
    image = io.imread('images/fur_obs.jpg', as_gray=True)
    # adjust pixel range
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = transform.resize(image, (im_w, im_h), mode='symmetric', preserve_range=True)
    image = (image * max_intensity).astype(np.int32).flatten()

    use_num = num_filters
    bin_num = 15
    threshold = 200
    weights = np.zeros((use_num, bin_num))
    unchosen_idx = list(range(use_num))
    chosen_idx = []
    avg_error_per_bin = []
    weighted_error_per_bin = []
    w = [weights]

    while len(chosen_idx) < use_num:
        err = []
        # select filter
        for i in unchosen_idx:
            im_res, min_res, max_res = get_histogram(lib, image, synthesized, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, 0.0, 0.0, 0)
            syn_res, _, _ = get_histogram(lib, synthesized, image, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, min_res, max_res, 1)
            err.append(np.sum(abs(im_res[0] - syn_res[0])))

        if np.max(err) < threshold:
            break
        idx = np.argmax(err)
        idx = unchosen_idx[idx]
        chosen_idx.append(idx)
        unchosen_idx.remove(idx)

        # update weights
        for i in chosen_idx:
            im_res, min_res, max_res = get_histogram(lib, image, synthesized, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, 0.0, 0.0, 0)
            syn_res, _, _ = get_histogram(lib, synthesized, image, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, min_res, max_res, 1)
            weights[i,:] = weights[i] + (syn_res[0] - im_res[0]) / im_w / im_h
        w.append(weights)

        # gibbs sampling
        synthesized, final_res = gibbs(lib, image, synthesized, filters[chosen_idx], width[chosen_idx], height[chosen_idx], im_w, im_h, bin_num, weights[chosen_idx,:].flatten(), max_intensity)

        err = []
        weighted_err = []
        # calculate error
        for i in chosen_idx:
            im_res, min_res, max_res = get_histogram(lib, image, synthesized, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, 0.0, 0.0, 0)
            syn_res, _, _ = get_histogram(lib, synthesized, image, np.array([filters[i]]), np.array([width[i]]), np.array([height[i]]), im_w, im_h, bin_num, min_res, max_res, 1)
            err.append(np.sum(abs(im_res[0] - syn_res[0])))
            weighted_err.append(np.sum(abs(im_res[0] - syn_res[0]) * weights[i]))
        avg_error_per_bin.append(np.mean(err) / bin_num)
        weighted_error_per_bin.append(np.mean(weighted_err) / bin_num)
        print("ave error per bin: %s\n" % avg_error_per_bin)


if __name__ == '__main__':
    main()
