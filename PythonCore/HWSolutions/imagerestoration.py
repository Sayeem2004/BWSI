import numpy as np;
import matplotlib.pyplot as plt
from bwsi_grader.python.image_restoration import grader1;
from bwsi_grader.python.image_restoration import grader2;
from bwsi_grader.python.image_restoration import grader3;

def compute_energy(img):
    ret = np.zeros_like(img);
    ret[:-1,:] += (img[:-1,:]!=img[1:,:]);
    ret[1:,:] += (img[1:,:]!=img[:-1,:]);
    ret[:,:-1] += (img[:,:-1]!=img[:,1:]);
    ret[:,1:] += (img[:,1:]!=img[:,:-1]);
    return ret;
grader1(compute_energy);

def get_neighbor_colors(img, pixel):
    ret = [];
    r,c = pixel;
    m,n = np.shape(img);
    if (r > 0): ret.append(img[r-1,c]);
    if (r < m-1): ret.append(img[r+1,c]);
    if (c > 0): ret.append(img[r,c-1]);
    if (c < n-1): ret.append(img[r,c+1]);
    return ret;
grader2(get_neighbor_colors);

def denoise_iter(noisy):
    energies = compute_energy(noisy);
    M,N = np.shape(energies);
    cell = np.argmax(energies);
    R,C = cell//N,cell%N
    neighbors = get_neighbor_colors(noisy,(R,C));
    values, counts = np.unique(neighbors,return_counts=True);
    ret = np.copy(noisy);
    ret[R,C] = values[np.argmax(counts)];
    return ret;
grader3(denoise_iter);

def generate_noisy_copy(img, pct_noise):
    noise = np.random.choice(np.unique(img), img.shape).astype(np.uint8);
    rands = np.random.rand(img.size).reshape(img.shape);
    noisy = img.copy();
    idxs_to_change = np.where(rands < pct_noise);
    noisy[idxs_to_change] = noise[idxs_to_change];
    return noisy;

# Loading in original image and generating a noisy copy.
pristine = (plt.imread('original-image.png')*255).astype(np.uint8);
noisy = generate_noisy_copy(pristine, 0.1);

# Cleaning up the noisy copy.
num_iters = 0                    # how many iterations we have performed, to see progress
cleaned_up = noisy.copy()        # the denoised image
old = np.zeros_like(cleaned_up)  # the previous iteration, for a stopping condition
while np.any(old != cleaned_up): # loop until no labels change values
    num_iters += 1
    if (num_iters%1000) == 0:    # print progress
        print(num_iters, 'Energy {}'.format(compute_energy(cleaned_up).sum()))
    old = cleaned_up.copy()
    cleaned_up = denoise_iter(cleaned_up)

# Visualizing
fig, axs = plt.subplots(1, 2, figsize=(8, 5));
axs[0].imshow(pristine, 'gray');
axs[1].imshow(noisy, 'gray');
fig, axs = plt.subplots(1, 3, figsize=(8, 3))
axs[0].imshow(noisy, 'gray')
axs[1].imshow(cleaned_up, 'gray')
axs[2].imshow(pristine, 'gray')
plt.show();
compute_energy(cleaned_up).sum()
compute_energy(pristine).sum()
