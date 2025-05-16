from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

image1 = Image.open('image1.jpg')
image2 = Image.open('image2.jpg')

transform = transforms.ToTensor()

image_tensor1 = transform(image1)
image_tensor2 = transform(image2)

plt.imshow(image_tensor1.permute(1,2,0))
plt.show()
plt.imshow(image_tensor2.permute(1,2,0))
plt.show()

print(image_tensor1.shape)
print(image_tensor1.dtype)
print(image_tensor2.shape)
print(image_tensor2.dtype)

res1 = image_tensor1+image_tensor2

plt.imshow(res1.permute(1,2,0))
plt.show()

res2 = image_tensor1.matmul(image_tensor2.permute(0,2,1))
print(f"the matrix multiplication of image1 and image2 is \n{res2}")


