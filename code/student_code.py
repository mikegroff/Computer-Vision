import numpy as np
from time import time

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  m,n,c = np.shape(image)
  j,k = np.shape(filter)
  a = j//2
  b = k//2

  for i in range(0,c):
      im = image[:,:,i]
      im = np.pad(im, ((a, a), (b, b)), 'reflect', reflect_type='odd')
      len,wid = np.shape(im)
      for l in range(a,len-a):
        for w in range(b, wid-b):
            block = im[l-a:l+a+1,w-b:w+b+1]
            block = block*filter
            im[l,w] = np.sum(block)
      image[:,:,i] = im[a:(m+a),b:(n+b)]
  filtered_image = image


  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """
  t1=time()
  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]
  m,n,c = np.shape(image2)
  j,k = np.shape(filter)

  filterh = np.zeros((j,k))
  j = j//2
  k = k//2
  filterh[j,k] = 1
  filterh = np.subtract(filterh, filter)

  imagea = my_imfilter(image1, filter)
  imageb = my_imfilter(image2, filterh)

  low_frequencies = imagea
  high_frequencies = imageb
  hybrid_image = np.clip((low_frequencies+high_frequencies),0,1)
  t2=time()
  Runtime=t2-t1
  #print(Runtime)

  return low_frequencies, high_frequencies, hybrid_image
