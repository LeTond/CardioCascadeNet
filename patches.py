import torch
import numpy as np


# # x = torch.randn(1, 500, 500, 500)  # batch, c, h, w
x = torch.randn(1, 1, 192, 192)  # batch, c, h, w
kc, kh, kw = 1, 64, 64  # kernel size
dc, dh, dw = 1, 32, 32  # stride

patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
patches2 = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
unfold_shape = patches.size()
patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
patches2 = patches2.contiguous().view(patches2.size(0), -1, kc, kh, kw)	#[1, 16, 1, 128, 128])
# print(f'Patch after unfolding: {patches[0,:,:,:,:].shape}')
print(f'Patches after unfolding: {patches.shape}')

# Reshape back
print(unfold_shape)
# patches_orig = patches.view(unfold_shape)
# patches_orig = patches.view(torch.Size([1, 3, 4, 4, 1, 128, 128]))

patches = np.array(patches, dtype = np.float32)
patches2 = np.array(patches2, dtype = np.float32)
print(f'Patches after np.array: {patches.shape}')

# patches = patches.transponse(0, 2, 1, 3, 4)
# patches2 = patches2.transponse(0, 2, 1, 3, 4)

patches3 = np.array([patches, patches2], dtype = np.float32)
patches3 = patches3.transpose(1, 3, 2, 0, 4, 5)
patches3 = torch.from_numpy(patches3)[:, :, :, :, :]

# unfold_shape = patches3.size()
print(f'Patches after merging: {patches3.shape}')

# unfold_shape = torch.Size([1, 1, 4, 4, 2, 128, 128])
unfold_shape = torch.Size([1, 1, 5, 5, 2, 64, 64])
patches_orig = patches3.view(unfold_shape)

output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]

patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)

# Check for equality
print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())
# print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())

print(patches_orig.shape)
print(patches[0, 0, 0, 20:64, 20:64])



# x = torch.randn(1, 500, 500, 500)  # batch, c, h, w
# x = torch.randn(1, 2, 512, 512)  # batch, c, h, w
# x = torch.randn(1, 512, 512)  # batch, c, h, w
# kh, kw = 128, 128  # kernel size
# dh, dw = 128, 128  # stride

# patches = x.unfold(1, kh, dh).unfold(2, kw, dw)
# unfold_shape = patches.size()
# patches = patches.contiguous().view(patches.size(0), -1, kh, kw)
# # print(patches[0, :, :, :].shape)

# # Reshape back

# print(patches.shape, unfold_shape)
# patches_orig = patches.view(unfold_shape)
# output_h = unfold_shape[1] * unfold_shape[3]
# output_w = unfold_shape[2] * unfold_shape[4]
# # output_w = unfold_shape[3] * unfold_shape[6]

# patches_orig = patches_orig.permute(0, 1, 3, 2, 4).contiguous()
# patches_orig = patches_orig.view(1, output_h, output_w)

# print(patches_orig.shape)

# # Check for equality
# print((patches_orig == x[:, :output_h, :output_w]).all())


# print(patches[0, 0, 20:64, 20:64])