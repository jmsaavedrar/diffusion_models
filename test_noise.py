import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x) :
    return 1.0 / (1 + np.exp((-x)))

def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.)

def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)

def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = np.cos(start * math.pi / 2) ** (2 * tau)
    v_end = np.cos(end * math.pi / 2) ** (2 * tau)    
    output = np.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.0)
               

def add_noise(image, noise_schedule, n):
    noisy_images = []
    image = image.astype(np.float32)
    image = (image / 255.0 )        
    
    tpos = np.linspace(0, len(noise_schedule) - 1, n).astype(int)
    for t in tpos:
        e = np.random.normal(0,1, image.shape)
        noisy = image * np.sqrt(noise_schedule[t]) + np.sqrt(1 - noise_schedule[t])*e        
        noisy_images.append(noisy)        
    return noisy_images


def logSNR(gamma) :
    return np.log(gamma / (1 - gamma))
    
if __name__ == '__main__':

    t = np.linspace(0,1.0, 100)
    gamma1 = cosine_schedule(t)
    gamma2 = sigmoid_schedule(t)
    # fig, xs = plt.subplots(1,2)
    # xs[0].plot(t, gamma1, 'red')    
    # xs[0].plot(t, gamma2, 'blue')
    # xs[1].plot(t, logSNR(gamma1), 'red')    
    # xs[1].plot(t, logSNR(gamma2), 'blue')
    # xs[0].set_title('Noise Schedule ')
    # xs[0].set_xlabel('t')
    # xs[0].set_ylabel('$\gamma$')
    # xs[1].set_title('logSNR')
    # xs[1].set_xlabel('t')
    # xs[1].set_ylabel('$\log(\gamma / (1 - \gamma))$')
    # plt.show()
    
    # #filename = '/home/vision/smb-datasets/Shoes/images_train/shoes/1185222993.png'
    filename = '/home/vision/smb-datasets/apple.jpg'
    image = io.imread(filename)
    n = 10
    noisy_images = add_noise(image, gamma2, n)
    _, xs = plt.subplots(1,n + 1)    
    for i in range(n + 1) :
        xs[i].set_axis_off()
    
    xs[0].imshow(image)
    for i in range(1, n) : 
        noisy_image = noisy_images[i]
        noisy_image = np.maximum(np.minimum(noisy_image * 255, 255),0)
        noisy_image =  noisy_image.astype(np.uint8)
        xs[i].imshow(noisy_image)       
        xs[i].set_title('t = {}'.format(i))        

    plt.show()
    