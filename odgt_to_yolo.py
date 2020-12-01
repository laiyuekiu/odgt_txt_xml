import json
import cv2
import imutils
import time
import os, glob

target_class = ['vbody'] #["head", "fbody", "vbody", #"mask"]   #[] --> all  ###you should pick one class only in yolo
annotations_path = "./annotation_train.odgt"
crowdHuman_path = "./crowd_human/all"


#output
datasetPath = 'yolo_crowd_dataset/'
imgPath = "image/"
labelPath = "label/"
imgType = ".png"  ### .jpg, .png (with "." dot)

def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))


def generateYOLO(filename, bbox):
    file = open(os.path.join(datasetPath, labelPath, filename+'.txt'), "w")
    file.writelines(bbox)
    file.close


def convert(size, box, img_id):
    if box[0] > size[0] or box[2] > size[0] or box[1] > size[1] or box[3] > size[1]:  ###wrong labelling, label coordinate is bigger than or overside the image size
        w = open(os.path.join(datasetPath, 'tagging_box_error'), 'a')
        w.write(img_id+'\n')
        w.close
        return [''] ###return nothing, give up this label
    dw = 1./(size[0])
    dh = 1./(size[1])
    xmax = abs(box[0]) + abs(box[2])
    ymax = abs(box[1]) + abs(box[3])
    if xmax > size[0]:  ###avoid the label box out of the image
        xmax = size[0]
    if ymax > size[1]:
        ymax = size[1]
    x = (abs(box[0]) + xmax)/2.0 - 1
    y = (abs(box[1]) + ymax)/2.0 - 1
    w = abs(box[2])
    h = abs(box[3])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [str(0), ' ', str(round(x, 6)),' ', str(round(y, 6)), ' ', str(round(w, 6)), ' ', str(round(h, 6)), '\n'] ###the first one, index 0, is the class id and assuem "person" class is 0


if __name__ == "__main__":
    check_env()

    f = open(annotations_path)
    lines = f.readlines()
    total_lines = len(lines)
    for lineID, line in enumerate(lines):
        data = eval(line)
        img_id = data["ID"]

        image_path = os.path.join(crowdHuman_path, img_id+'.jpg')
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            #cv2.imwrite(os.path.join(datasetPath, imgPath, img_id + imgType), img) ###uncomment to generate image
            (img_height, img_width, img_depth) = img.shape
            total_iid = len(data["gtboxes"])       
            box_list=[]

            for iid, infoBody in enumerate(data["gtboxes"]):
                if infoBody['tag'] != 'mask':
                    tag = infoBody["tag"]
                    hbox = infoBody["hbox"]
                    head_attr = infoBody["head_attr"]
                    fbox = infoBody["fbox"]
                    vbox = infoBody["vbox"]
                    extra = infoBody["extra"]
                    difficult = 0

                    if(len(target_class)==0 or ("head" in target_class)):
                        box_list.extend(convert((img_width, img_height), hbox, img_id))

                    if(len(target_class)==0 or ("fbody" in target_class)):
                        box_list.extend(convert((img_width, img_height), fbox, img_id))

                    if(len(target_class)==0 or ("vbody" in target_class)):
                        box_list.extend(convert((img_width, img_height), vbox, img_id))
            
                    print("[Left over] {}/{} , {}: head:{} full:{} visible:{}".format(total_lines-lineID, total_iid-iid, img_id,\
                        hbox, fbox, vbox)) #end of one label object loop 
            generateYOLO(img_id, box_list) #end of one image loop
        else:
            f = open(os.path.join(datasetPath, 'no_this_image_log'), 'a')  ###can't find the label file name in imgPath
            f.write(img_id+'\n')
            f.close
