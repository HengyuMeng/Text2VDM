import numpy as np
from PIL import Image
import torch
import torchvision

def generate_image_with_gradient_ellipse_and_mask(width=512, height=512, ellipse_size=(200, 100), inner_color=(255, 0, 0), edge_color=(0, 255, 0), gradient_color=(0, 0, 0), gradient_width=20, gradient_intensity=1.0):
    
    """
    Generate an Elliptical Image with Double Gradient Edges
    
    :param width: Width of the image
    :param height: Height of the image
    :param ellipse_size: Size of the ellipse (width, height)
    :param inner_color: Inner color of the ellipse (R, G, B)
    :param edge_color: Edge color of the ellipse (first gradient target color)
    :param gradient_color: Target color for the outer gradient of the ellipse (R, G, B)
    :param gradient_width: Width of the gradient (distance from the ellipse edge to the target color)
    :param gradient_intensity: Gradient intensity, the larger the value, the faster the gradient
    """
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    pixels = np.array(img)
    center_x = width // 2
    center_y = height // 2
    rx, ry = ellipse_size[0] // 2, ellipse_size[1] // 2
    
    for y in range(height):
        for x in range(width):
            normalized_distance = ((x - center_x) / rx) ** 2 + ((y - center_y) / ry) ** 2
            
            if normalized_distance <= 1:
                gradient_ratio = normalized_distance ** gradient_intensity
                current_color = tuple(
                    int(inner_color[i] * (1 - gradient_ratio) + edge_color[i] * gradient_ratio) 
                    for i in range(3)
                )
                pixels[y, x] = current_color
            elif 1 < normalized_distance <= (1 + gradient_width / max(rx, ry)):
                gradient_ratio = ((normalized_distance - 1) / (gradient_width / max(rx, ry))) ** gradient_intensity
                current_color = tuple(
                    int(edge_color[i] * (1 - gradient_ratio) + gradient_color[i] * gradient_ratio) 
                    for i in range(3)
                )
                pixels[y, x] = current_color
    
    img = Image.fromarray(pixels)

    return img


size = (290, 160)
center_color = (0, 0, 150)
edge_color = (0, 5, 5)
width = 40
intensity = 1.5

image = generate_image_with_gradient_ellipse_and_mask(ellipse_size=size, inner_color=center_color, edge_color=edge_color, gradient_color=(0, 0, 0), gradient_width=width, gradient_intensity=intensity)
image.save('shape_image.png')

mask = generate_image_with_gradient_ellipse_and_mask(ellipse_size=size, inner_color=(255,255,255), edge_color=(0,0,0), gradient_color=(0, 0, 0), gradient_width=width, gradient_intensity=100)
mask.save('shape_mask.png')