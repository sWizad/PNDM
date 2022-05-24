import os

#@title Generate
dataset = 'cifar10'
if True:
    config = 'ddim_cifar10.yml'
    model_path = 'pretrained/models/ddim_cifar10.ckpt'
elif True:
    config = 'iddpm_cifar10.yml'
    model_path = 'pretrained/models/iddpm_cifar10.ckpt'
outdir = 'pretrained/results/test' #@param {type: 'string'}
method = "NEWT4n4"#"F-PNDM" #PNDM2
#step_list = [12, 25]
#step_list = [50, 100, 200]
step_list = [ 12, 25, 50, 100, 200]
#step_list = [ 4, 5]

for step in step_list: # 
    #outdir = os.path.join('pretrained/results/',dataset,method,str(step))
    outdir = os.path.join('pretrained/results/',dataset,'imp_dgdt',method,str(step))
    cmd = f'python main.py --runner sample ' \
          f'--method {method} --sample_speed {step} --device cuda '\
          f'--config {config} --model_path {model_path}  '\
          f'--image_path {outdir}'

    print(cmd)
    os.system(cmd)