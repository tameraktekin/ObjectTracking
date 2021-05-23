from configparser import ConfigParser
config = ConfigParser()

config.add_section('data')
config.set('data', 'test_folder', './test/')
config.set('data', 'test_video', '22.mp4')
config.set('data', 'width', '1280')
config.set('data', 'height', '720')

config.add_section('model')
config.set('model', 'model_folder', './model/')
config.set('model', 'model_proto_name', 'MobileNetSSD_deploy.prototxt.txt')
config.set('model', 'model_name', 'MobileNetSSD_deploy.caffemodel')

config.add_section('test')
config.set('test', 'save_video', 'True')
config.set('test', 'output_file', 'test_output.avi')
config.set('test', 'output_fps', '10')

with open('config.ini', 'w') as f:
    config.write(f)
