import json
import cv2
import imutils
import time
import os, glob
 
target_class = [] #["head", "fbody", "vbody", #"mask"]   #[] --> all
annotations_path = "./annotation_train.odgt"
crowdHuman_path = "./crowd_human/all"


xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = 'crowd_dataset/'
imgPath = "image/"
labelPath = "label/"
imgType = ".png"  # .jpg, .png (with "." dot)

def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))


def generateXML(filename, bbox, img_height, img_width, img_depth):
    folder_name = 'your_folder_name'
    file_name = filename + imgType
    database_name = 'your_database'
    segmented = 0
    file_path = os.path.join(datasetPath, imgPath, file_name)

    xmlObject = ['<annotation>\n', 
                    '\t<folder>', str(folder_name), '</folder>\n',
                    '\t<filename>', str(file_name), '</filename>\n',
                    '\t<path>', str(file_path), '</path>\n',
                    '\t<source>\n', '\t\t<database>', str(database_name), '</database>\n', '\t</source>\n',
                    '\t<size>\n', '\t\t<width>', str(abs(img_width)), '</width>\n', '\t\t<height>', str(abs(img_height)), '</height>\n', '\t\t<depth>', str(abs(img_depth)), '</depth>\n', '\t</size>\n',
                    '\t<segmented>', str(segmented), '</segmented>',                               
        ]

    xmlObject.extend(bbox)
    xmlObject.append('\n</annotation>')

    xml_file = filename + '.xml'
    file = open(os.path.join(datasetPath, labelPath, xml_file), "w")
    file.writelines(xmlObject)
    file.close


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
                    box_list.extend(['\n\t<object>\n'])
                    tag = infoBody["tag"]
                    hbox = infoBody["hbox"]
                    head_attr = infoBody["head_attr"]
                    fbox = infoBody["fbox"]
                    vbox = infoBody["vbox"]
                    extra = infoBody["extra"]
                    difficult = 0
                    box_list.extend(['\t\t<name>', str(tag), '</name>\n', '\t\t<pose>Unspecified</pose>\n', 
                                     '\t\t<truncated>0</truncated>\n', '\t\t<difficult>', str(difficult), '</difficult>\n'
                                     ])

                    if(len(target_class)==0 or ("head" in target_class)):
                        ###xmin,ymin,xmax,ymax need in positive value and smaller than image size, so the bndbox may not really fit
                        h_xmax = abs(hbox[0])+abs(hbox[2])
                        h_ymax = abs(hbox[1])+abs(hbox[3])
                        if abs(hbox[0]) > img_width:
                            hbox[0] = img_width
                        if h_xmax > img_width:
                            h_xmax = img_width
                        if abs(hbox[1]) > img_height:
                            hbox[1] = img_height
                        if h_ymax > img_height:
                            h_ymax = img_height
                        
                        box_list.extend(['\t\t<hbox>\n', '\t\t\t<bndbox>\n', 
                                         '\t\t\t\t<xmin>', str(abs(hbox[0])), '</xmin>\n', '\t\t\t\t<ymin>', str(abs(hbox[1])), '</ymin>\n', 
                                         '\t\t\t\t<xmax>', str(h_xmax), '</xmax>\n', '\t\t\t\t<ymax>', str(h_ymax), '</ymax>\n', 
                                         '\t\t\t</bndbox>\n', '\t\t</hbox>\n'
                            ])

                    if(len(target_class)==0 or ("fbody" in target_class)):
                        f_xmax = abs(fbox[0])+abs(fbox[2])
                        f_ymax = abs(fbox[1])+abs(fbox[3])
                        if abs(fbox[0]) > img_width:
                            fbox[0] = img_width
                        if f_xmax > img_width:
                            f_xmax = img_width
                        if abs(fbox[1]) > img_height:
                            fbox[1] = img_height
                        if f_ymax > img_height:
                            f_ymax = img_height

                        box_list.extend(['\t\t<fbox>\n', '\t\t\t<bndbox>\n', 
                                         '\t\t\t\t<xmin>', str(abs(fbox[0])), '</xmin>\n', '\t\t\t\t<ymin>', str(abs(fbox[1])), '</ymin>\n', 
                                         '\t\t\t\t<xmax>', str(f_xmax), '</xmax>\n', '\t\t\t\t<ymax>', str(f_ymax), '</ymax>\n', 
                                         '\t\t\t</bndbox>\n', '\t\t</fbox>\n'
                            ])

                    if(len(target_class)==0 or ("vbody" in target_class)):
                        v_xmax = abs(vbox[0])+abs(vbox[2])
                        v_ymax = abs(vbox[1])+abs(vbox[3])
                        if abs(vbox[0]) > img_width:
                            vbox[0] = img_width
                        if v_xmax > img_width:
                            v_xmax = img_width
                        if abs(vbox[1]) > img_height:
                            vbox[1] = img_height
                        if v_ymax > img_height:
                            v_ymax = img_height

                        box_list.extend(['\t\t<vbox>\n', '\t\t\t<bndbox> \n', 
                                         '\t\t\t\t<xmin>', str(abs(vbox[0])), '</xmin>\n', '\t\t\t\t<ymin>', str(abs(vbox[1])), '</ymin>\n', 
                                         '\t\t\t\t<xmax>', str(v_xmax), '</xmax>\n', '\t\t\t\t<ymax>', str(v_ymax), '</ymax>\n', 
                                         '\t\t\t</bndbox>\n', '\t\t</vbox>\n'
                            ])

                    box_list.append('\t</object>') #end of one object loop                
                    print("[Left over] {}/{} , {}: head:{} full:{} visible:{}".format(total_lines-lineID, total_iid-iid, img_id,\
                        hbox, fbox, vbox))
            generateXML(img_id, box_list, img_height, img_width, img_depth) #end of one image loop
        else:
            f = open(os.path.join(datasetPath, 'no_this_image_log'), 'a')
            f.write(img_id+'\n')
            f.close 
