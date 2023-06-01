import xml.etree.ElementTree as ET
import os

#train_path = r'C:\Users\harit\Downloads\This sem\FYP\Code\FYP_v4\Tensorflow\workspace\images\train'
train_path = r'C:\Users\harit\Downloads\This sem\FYP\Code\FYP_v4\Tensorflow\workspace\images\test'
ext = ('.xml')

def xmlNameUpdater(filepath, filename):
    tree = ET.ElementTree(file=filepath)
    root = tree.getroot()
    for name in root.iter('filename'):
        name.text = filename.split('.')[0] + '.jpg'

    tree = ET.ElementTree(root) #update tree
    with open(filepath, 'wb') as fileupdate:
        tree.write(fileupdate)

def xmlDeleteObject(filepath):
    tree = ET.ElementTree(file=filepath)
    root = tree.getroot()
    for traffic_object in root.findall('object'):
        object_name = traffic_object.find('name')
        if object_name.text == 'traffic sign': root.remove(traffic_object)
        if object_name.text == 'traffic light': root.remove(traffic_object)
        if object_name.text == 'none': root.remove(traffic_object)

    tree = ET.ElementTree(root)  # update tree
    with open(filepath, 'wb') as fileupdate:
        tree.write(fileupdate)

for path, dirc, files in os.walk(train_path):
    for name in files:
        if name.endswith(ext):
            filepath = os.path.join(train_path, name)
            xmlNameUpdater(filepath, name)
            xmlDeleteObject(filepath)