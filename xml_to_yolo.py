import os, glob, xml.etree.ElementTree as ET

splits = ['train','valid','test']
xml_paths = []
for split in splits:
    xml_paths += glob.glob(f"data/{split}/*.xml")

classes = set()
for xp in xml_paths:
    root = ET.parse(xp).getroot()
    for obj in root.findall('object'):
        classes.add(obj.find('name').text)
classes = sorted(classes)
print("Using classes:", classes)

class_to_id = {c:i for i,c in enumerate(classes)}


for split in splits:
    for xml_path in glob.glob(f"data/{split}/*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            cls_id = class_to_id.get(cls)
            if cls_id is None:
                continue  

            b = obj.find('bndbox')
            xmin, ymin = float(b.find('xmin').text), float(b.find('ymin').text)
            xmax, ymax = float(b.find('xmax').text), float(b.find('ymax').text)


            x_c = ((xmin + xmax)/2) / w
            y_c = ((ymin + ymax)/2) / h
            bw  = (xmax - xmin) / w
            bh  = (ymax - ymin) / h

            yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")


        txt_path = xml_path[:-4] + '.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))