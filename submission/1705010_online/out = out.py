out = out.reshape(F, H_out, W_out, N)
out = out.transpose(3, 0, 1, 2)