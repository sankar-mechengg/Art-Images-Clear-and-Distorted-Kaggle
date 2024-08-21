# Create Distortions in Art Images

# Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import time
import copy
from skimage.transform import swirl

####################################################################################################

# Read Image from a folder


def read_image(path):
    file_name = os.path.basename(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


####################################################################################################

# Resize Image to 512 x 512


def resize_image(image):
    image = cv2.resize(image, (512, 512))
    return image


####################################################################################################

# Save Images from a list to a folder


def save_images(image_list, image_name):
    for i in range(len(image_list)):
        path = r"H:\Code\Python\Dataset\Distorted"
        image = image_list[i]
        image = np.float32(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
        # Save Image using a unique name
        file_name = image_name + "_" + str(i) + ".jpg"
        path = os.path.join(path, file_name)
        cv2.imwrite(path, image)


####################################################################################################

# Gaussian & Motion Blur


def apply_blur(image, kernel_size):
    image = cv2.GaussianBlur(image, kernel_size, 0)
    return image


####################################################################################################

# Gaussian Noise


def apply_gaussian_noise(image, mean, stdev):
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, stdev)
    image = cv2.add(image, noise)
    return image


####################################################################################################

# Speckle Noise


def apply_speckle_noise(image, speckle_value):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    speckle_noisy_image = image + image * speckle_value * gauss
    speckle_noisy_image = speckle_noisy_image.astype(np.uint8)
    return speckle_noisy_image


####################################################################################################

# Fading


def apply_fading(image, desaturation_percent, devalue_percent):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Split the channels
    img_h_channel = copy.deepcopy(img_hsv[:, :, 0])
    img_s_channel = copy.deepcopy(img_hsv[:, :, 1])
    img_v_channel = copy.deepcopy(img_hsv[:, :, 2])
    # Apply Fading
    faded_s_channel = (img_s_channel.astype(np.float64) * desaturation_percent).astype(
        np.uint8
    )
    faded_v_channel = (img_v_channel.astype(np.float64) * devalue_percent).astype(
        np.uint8
    )
    # Merge the channels
    faded_img_hsv = cv2.merge([img_h_channel, faded_s_channel, faded_v_channel])
    faded_image_rgb = cv2.cvtColor(faded_img_hsv, cv2.COLOR_HSV2RGB)
    return faded_image_rgb


####################################################################################################

# White Overlay and Fading


def apply_white_overlay_and_fading(image, desaturation_percent, devalue_percent):
    # Apply Fading
    faded_image = apply_fading(image, desaturation_percent, devalue_percent)
    # Create a white overlay
    white_overlay = np.ones_like(faded_image) * 255
    opacity = 0.6
    # Merge the white overlay and the image
    white_overlay_faded_image = cv2.addWeighted(
        faded_image, 1 - opacity, white_overlay, opacity, 0
    )
    return white_overlay_faded_image


####################################################################################################

# Scratches


def apply_scratches(image, scratch_texture, intensity):
    scratch_texture = cv2.resize(scratch_texture, (512, 512))
    # Convert Scratch Texture to BW Inverse
    scratch_texture = cv2.cvtColor(scratch_texture, cv2.COLOR_BGR2GRAY)
    scratch_texture = np.uint8(scratch_texture)

    (thresh, scratch) = cv2.threshold(
        scratch_texture, intensity, 255, cv2.THRESH_BINARY
    )
    scratch = cv2.bitwise_not(scratch)
    image = cv2.bitwise_and(image, image, mask=scratch)
    return image


####################################################################################################

# Swirl


def apply_swirl(image, strength, radius, rotation):
    motion_blur_image = apply_blur(image, kernel_size=(5, 15))
    image = np.float32(motion_blur_image)
    swirled_image = swirl(
        image, strength=strength, radius=radius, rotation=rotation
    ).astype(np.uint8)
    return swirled_image


####################################################################################################

# Water Discoloration


def apply_water_discoloration(image, sigma_s, sigma_r):
    watercolour_image = cv2.edgePreservingFilter(
        image, sigma_s=sigma_s, sigma_r=sigma_r, flags=2
    )
    return watercolour_image


####################################################################################################

# Pixelation


def apply_pixelation(image, pixel_size):
    height, width = image.shape[:2]
    # Resize input to "pixelated" size
    temp = cv2.resize(image, pixel_size, interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    pixelated_image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image


####################################################################################################

# Darken Image


def apply_darken(image, darken_value):
    M = np.ones(image.shape, dtype="uint8") * darken_value  # Matrix M
    # Subtract M from image
    subtracted_image = cv2.subtract(image, M)
    return subtracted_image


####################################################################################################

# Vertical Scratch


def apply_vertical_scratch(image, scratch_texture_v, intensity):
    scratch_texture_v = cv2.resize(scratch_texture_v, (512, 512))
    # Convert Scratch Texture to BW Inverse
    scratch_texture_v = cv2.cvtColor(scratch_texture_v, cv2.COLOR_BGR2GRAY)

    # Replace Pixel Values
    def scratch_pixel_image(image_ip, scratch):
        for i in range(image_ip.shape[0]):
            for j in range(image_ip.shape[1]):
                if (scratch[i][j] != 0).all():
                    image_ip[i][j] = scratch[i][j]
        return image_ip

    (thresh, scratch) = cv2.threshold(
        scratch_texture_v, intensity, 255, cv2.THRESH_BINARY
    )
    # scratch = cv2.bitwise_not(scratch)

    # # Taking each layer and Taking mean of each layer and Multiplying with scratch and Replace Pixel Values
    k_b, k_g, k_r = cv2.split(image)

    b = scratch_pixel_image(k_b, scratch)
    g = scratch_pixel_image(k_g, scratch)
    r = scratch_pixel_image(k_r, scratch)

    scratched_image = cv2.merge([b, g, r])
    return scratched_image


####################################################################################################


# Vertical Scratch using Masking


def apply_vertical_scratch_mask(image, scratch_texture_v, intensity):
    scratch_texture_v = cv2.resize(scratch_texture_v, (512, 512))
    # Convert Scratch Texture to BW Inverse
    scratch_texture_v = cv2.cvtColor(scratch_texture_v, cv2.COLOR_BGR2GRAY)
    scratch_texture_v = np.uint8(scratch_texture_v)

    (thresh, scratch) = cv2.threshold(
        scratch_texture_v, intensity, 255, cv2.THRESH_BINARY
    )
    mask = scratch

    scratched_image = cv2.bitwise_not(image, image, mask=mask)
    del scratch_texture_v, scratch, mask

    return scratched_image


####################################################################################################

# Apply All Distortion Function


def apply_all_distortion(path, file_name):
    img = read_image(path)
    img = resize_image(img)
    gaussian_blur = 1
    motion_blur = 1
    gaussian_noise = 1
    speckle_noise = 1
    fading = 1
    white_overlay_fading = 1
    swirl = 1
    water_discoloration = 1
    pixelation = 1
    darken_image = 1
    scratches = 1
    vertical_scratch = 1

    # Distorted Image List
    distorted_image_list = []

    # Gaussian Blur
    # kernel size (x,y) has to be odd and bigger the number, more the blur. Note that for gaussian blur x must equal to y
    if gaussian_blur == True:
        kernel_sizes = [(5, 5), (15, 15), (25, 25), (35, 35), (45, 45)]
        for size in kernel_sizes:
            img_gauss_blur = apply_blur(img, size)
            distorted_image_list.append(img_gauss_blur)
        # print("Gaussian Blur Done")

    # Motion Blur
    # kernel size (x,y) has to be odd and the magnitude of motion blur depends on the positive difference of kernel size (x,y). Note that for motion blur x not equal to y. For eg: (5,15) --> motion blur in y direction
    if motion_blur == True:
        kernel_sizes = [
            (5, 15),
            (5, 25),
            (5, 35),
            (5, 45),
            (15, 5),
            (25, 5),
            (35, 5),
            (45, 5),
        ]
        for size in kernel_sizes:
            img_motion_blur = apply_blur(img, size)
            distorted_image_list.append(img_motion_blur)
        # print("Motion Blur Done")

    # Gaussian Noise
    if gaussian_noise == True:
        mean = 0
        stdev_list = [50, 150, 210, 270]
        for stdev in stdev_list:
            img_gauss_noise = apply_gaussian_noise(img, mean, stdev)
            distorted_image_list.append(img_gauss_noise)
        # print("Gaussian Noise Done")

    # Speckle Noise
    if speckle_noise == True:
        speckle_value_list = [0.2, 0.4, 0.6, 0.8]
        for speckle_value in speckle_value_list:
            img_speckle_noise = apply_speckle_noise(img, speckle_value)
            distorted_image_list.append(img_speckle_noise)
        # print("Speckle Noise Done")

    # Fading
    # lesser the number, lesser the pigment (i.e., more fading)
    if fading == True:
        desaturation_percent_list = [0.5]
        # lesser the number, blacker the image (if devalue_percentage=1 and desaturation_percentage=1, edited_image = original_image)
        devalue_percent_list = [0.2, 0.4, 0.6, 0.8]
        for desaturation_percent in desaturation_percent_list:
            for devalue_percent in devalue_percent_list:
                img_fading = apply_fading(img, desaturation_percent, devalue_percent)
                distorted_image_list.append(img_fading)
        # print("Fading Done")

    # White Overlay and Fading
    # lesser the number, lesser the pigment (i.e., more fading)
    if white_overlay_fading == True:
        desaturation_percent_list = [0.1, 0.3, 0.6, 0.9, 1.2]
        devalue_percent = 1
        for desaturation_percent in desaturation_percent_list:
            img_white_overlay_fading = apply_white_overlay_and_fading(
                img, desaturation_percent, devalue_percent
            )
            distorted_image_list.append(img_white_overlay_fading)
        # print("White Overlay and Fading Done")

    # Swirl
    if swirl == True:
        rotation = 0
        radius = 1500
        # higher the strength, higher the effect of swirl distortion; lesser the radius, swirl distortion is more concentrated on the center
        strength_list = [0.5, 1, 1.5, 2, 2.5]
        for strength in strength_list:
            img_swirl = apply_swirl(img, strength, radius, rotation)
            distorted_image_list.append(img_swirl)
        # print("Swirl Done")

    # Water Discoloration
    if water_discoloration == True:
        sigma_s = 40
        sigma_r = 0.7
        # higher the sigma_s, more the effect of watercolorization
        sigma_s_values = [40]
        # higher the sigma_r, more the effect of watercolorization
        sigma_r_values = [0.3, 0.5, 0.7, 0.9]
        # Thus, let's set sigma_s=40, run over the values of sigma_r
        for sigma_r_value in sigma_r_values:
            for sigma_s_value in sigma_s_values:
                img_water_discoloration = apply_water_discoloration(
                    img, sigma_s_value, sigma_r_value
                )
                distorted_image_list.append(img_water_discoloration)
        # print("Water Discoloration Done")

    # Pixelation
    if pixelation == True:
        pixel_size_list = [(50, 50), (75, 75), (100, 100), (125, 125)]
        for pixel_size in pixel_size_list:
            img_pixelation = apply_pixelation(img, pixel_size)
            distorted_image_list.append(img_pixelation)
        # print("Pixelation Done")

    # Darken Image
    if darken_image == True:
        darken_value_list = [50, 100, 150]
        for darken_value in darken_value_list:
            img_darken = apply_darken(img, darken_value)
            distorted_image_list.append(img_darken)
        # print("Darken Image Done")

    # Scratches
    if scratches == True:
        scratch_texture = read_image(
            r"H:\Code\Python\OpenCV\Dataset\Textures\Multi_Scratch_Texture.jpeg"
        )
        scratch_intensity_list = [25, 50, 100]
        for scratch_intensity in scratch_intensity_list:
            img_scratches = apply_scratches(img, scratch_texture, scratch_intensity)
            distorted_image_list.append(img_scratches)
        # print("Scratches Done")

    # Vertical Scratch
    if vertical_scratch == True:
        scratch_texture_vertical = read_image(
            r"H:\Code\Python\OpenCV\Dataset\Textures\Verti_Scratch_Texture_2.jpeg"
        )
        img_vertical_scratch = apply_vertical_scratch_mask(
            img, scratch_texture_vertical, 50
        )
        distorted_image_list.append(img_vertical_scratch)
        # print("Vertical Scratch Done")

    # Save Distorted Images
    # print("Total Images:" + str(len(distorted_image_list)))
    save_images(distorted_image_list, file_name)
    return


####################################################################################################

# Main Function
if __name__ == "__main__":
    # Path to the image
    path = r"H:\Code\Python\OpenCV\Dataset\Test"
    begin = time.time()
    for images in os.listdir(path):
        path = os.path.join(path, images)
        file_name = os.path.basename(path)
        # print(file_name)
        # strip the file extension
        file_name = os.path.splitext(file_name)[0]
        apply_all_distortion(path, file_name)  # Call Main Function
    end = time.time()
    print("Time Taken: " + str(end - begin))
