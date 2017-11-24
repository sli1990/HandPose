from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
#checkpoint_file = os.path.join('C:/Users\Jan\Desktop\Read_CKPT', 'model.ckpt-2812499')
checkpoint_file = os.path.join('C:/Users\Jan\Desktop\Read_CKPT', 'model.ckpt-0')
#checkpoint_path = os.path.join(model_dir, "model.ckpt")

# List ALL tensors
print_tensors_in_checkpoint_file(file_name=checkpoint_file, tensor_name='', all_tensors=True)