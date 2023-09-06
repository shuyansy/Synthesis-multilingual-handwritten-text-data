import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import numpy as np
import random 
import os



def main(mask_list_file,data,lan,vis):
    output_path = os.path.join(lan, "images")
    vis_path = os.path.join(lan, "visual")

    # 判断子文件夹是否存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)


     # write into coco-style
    # COCO 数据格式的基本结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    for num in range(9):
        for i in range(len(data["documents"])):

            each_document= data["documents"][i]
            print("Document ID:", each_document["id"])
            print("Document UID:", each_document["uid"])

            image_path = "XFUND/images/"+ each_document["id"] + ".jpg"
            image = cv2.imread(image_path)
            h,w,c=image.shape[:]

            # 获取 "label" 为 "answer" 和 "question" 的框坐标
            answer_boxes = [item["box"] for item in each_document["document"] if item["label"] == "answer"]
            question_boxes = [item["box"] for item in each_document["document"] if item["label"] == "question"]

            bbox=[]
            # 在图像上绘制 "label" 为 "answer" 的框
            for box in answer_boxes:
                x1, y1, x2, y2 = box
                bbox.append([x1,y1,x2,y2])
                if vis:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 红色框

            # 在图像上绘制 "label" 为 "question" 的框
            for box in question_boxes:
                x1, y1, x2, y2 = box
                bbox.append([x1,y1,x2,y2])
                if vis:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框



            # 填充图片信息
            image_id =each_document["id"]+"_"+str(num) # 图像 ID，可以自行设置
            image_file_name = each_document["id"] +"_"+str(num) + ".jpg"  # 图像文件名
            image_width = w  # 图像宽度
            image_height = h  # 图像高度
            image_info = {
                "id": image_id,
                "file_name": image_file_name,
                "width": image_width,
                "height": image_height
            }
            coco_data["images"].append(image_info)

           

            i=0
            for b in bbox:
                x1,y1,x2,y2=b[0],b[1],b[2],b[3]
                area=(x2-x1) * (y2-y1)
                anno={"id":i,
                "image_id":image_id,
                "category_id":1,
                "bbox":[x1,y1,x2-x1,y2-y1],
                "area":area,
                "iscrowd":0,
                "segmentation":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                }
                i+=1

                # 填充标注信息
                coco_data["annotations"].append(anno)


            # random 50% mask
            num_boxes_to_mask = len(answer_boxes) // 2
            # 随机选择需要蒙版的框
            boxes_to_mask = random.sample(answer_boxes, num_boxes_to_mask)




            # 在图像上绘制蒙版
            # 在图像上绘制蒙版并贴上另一张图片
            for box in boxes_to_mask:
                choice_mask_path = random.sample(mask_list,1)
                overlay=cv2.imread(os.path.join(mask_list_file,choice_mask_path[0]))
                h_over,w_over,c=overlay.shape[:]

                x1, y1, x2, y2 = box
                w=x2-x1
                y=y2-y1
                if w_over> 2*w or w_over < 0.5*w:
                    continue


                masked_area = image[y1:y2, x1:x2, :]
                overlay_resized = cv2.resize(overlay, (x2 - x1, y2 - y1))
                masked_area[:] = overlay_resized  # 将另一张图片贴到蒙版区域
                


            # 保存绘制后的图像
            img_name=each_document["id"]+"_"+str(num)+".jpg"
            cv2.imwrite(os.path.join(output_path,img_name), image)

            if vis:
                cv2.imwrite(os.path.join(vis_path,img_name), image)

        num+=1
    

    # 将 COCO 数据写入 JSON 文件
     # 填充类别信息
    category_id_to_name = {1: "text"}  # 根据您的类别ID进行映射
    for category_id, category_name in category_id_to_name.items():
        category_info = {
            "id": category_id,
            "name": category_name,
            "supercategory": category_name  # 这里简化，将超类别设为类别名
        }
        coco_data["categories"].append(category_info)
    output_file = "annotations.json"
    with open(os.path.join(lan,output_file), "w") as f:
        json.dump(coco_data, f, indent=4)

    print("COCO annotations saved to:", output_file)





if __name__ == "__main__":
    mode=["en","Ja","ch"]
    if "en" in mode:
        lan="en"
        mask_list_file="en_hand_data/"
        mask_list=os.listdir(mask_list_file)
        annotation_file="XFUND/zh.train.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)


        annotation_file="XFUND/zh.val.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)

    if "Ja" in mode:
        lan="Ja"
        mask_list_file="Jan_hand_data/"
        mask_list=os.listdir(mask_list_file)
        annotation_file="XFUND/zh.train.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)


        annotation_file="XFUND/zh.val.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)
    
    if "ch" in mode:
        lan="ch"
        mask_list_file="corpus/"
        mask_list_1=os.listdir(mask_list_file)
        mask_list=[]
        for i in mask_list_1:
            file_1=os.listdir(os.path.join(mask_list_file,i))
            for f in file_1:
                mask_list.append(os.path.join(i,f))
        

        annotation_file="XFUND/zh.train.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)

        annotation_file="XFUND/zh.val.json"
        f = open(annotation_file, 'r')
        content = f.read()
        data = json.loads(content)
        main(mask_list_file,data,lan,vis=False)





